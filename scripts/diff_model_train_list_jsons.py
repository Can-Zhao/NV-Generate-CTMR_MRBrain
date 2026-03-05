# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced Diffusion Model Training with Multi-Dataset Support

Key Enhancements vs diff_model_train.py:
─────────────────────────────────────────────────────────────────────────────

1. Multi-Dataset Support: Train on multiple datasets simultaneously
2. Embedded Metadata: Spacing, modality, dim stored in JSON (no separate files)
3. Robust Checkpoints: Latest + periodic (every 200 epochs) with auto-fallback
4. Image Dimension Filtering: Filter by size for batch_size > 1
5. Weighted Sampling: Balance training across underrepresented modalities
6. TensorBoard: Real-time loss curves and validation image visualization
7. Validation Images: Auto-generate every VAL_INTERVAL epochs with XYZ view
8. YAML Support: JSON and YAML configs (auto-detected)
9. Auto-Distributed: Works with torchrun, no num_gpus argument needed
10. Bug Fixes: Fixed tensor ops and multi-node GPU handling

Checkpoints:
─────────────────────────────────────────────────────────────────────────────
- Latest: model.pt (every epoch, overwrites)
- Periodic: model_epoch{N}.pt (every 200 epochs, kept)
- Auto-fallback: If latest corrupted, loads most recent periodic
- Formats: .pt (single file) or .safetensors (model + training state)

Usage:
─────────────────────────────────────────────────────────────────────────────
Single dataset:
  json_data_list: "/path/to/dataset.json"
  embedding_base_dir: "/path/to/embeddings/"

Multiple datasets:
  json_data_list: ["/path1.json", "/path2.json", "/path3.json"]
  embedding_base_dir: ["/embeddings1/", "/embeddings2/", "/embeddings3/"]

Filter by dimension (for batch_size > 1):
  image_dim: [128, 128, 128]
  batch_size: 4

All dimensions (requires batch_size=1):
  image_dim: null
  batch_size: 1

When to Use:
─────────────────────────────────────────────────────────────────────────────
Use this script: Multi-dataset, weighted sampling, TensorBoard, validation
Use diff_model_train.py: Simple single-dataset, legacy compatibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance, load_checkpoint, load_unet, save_checkpoint
# Import inference functions directly (avoid distributed setup/teardown)
from .diff_model_infer import load_models, prepare_tensors, run_inference

# Checkpoint saving interval (save every N epochs)
SAVE_EPOCH_INTERVAL = 500

# Validation image generation interval (generate every N epochs)
VAL_INTERVAL = 250

def augment_modality_label(modality_tensor, prob=0.1):
    """
    Augments the modality tensor by randomly modifying certain elements based on a given probability.

    - A proportion of elements (determined by `prob`) are randomly set to 0.
    - Elements equal to 2 or 3 are randomly set to 1 with a probability defined by `prob`.
    - Elements between 9 and 12 are randomly set to 8 with a probability defined by `prob`.

    Parameters:
    modality_tensor (torch.Tensor): A tensor containing modality labels.
    prob (float): The probability of modifying certain elements (should be between 0 and 1).
                  For example, if `prob` is 0.3, there's a 30% chance of modification.

    Returns:
    torch.Tensor: The modified modality tensor with the applied augmentations.
    """
    # Randomly set elements that are smaller than 8 with probability `prob`
    mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1
    
    # Randomly set elements larger than 9 with probability `prob`
    mask_mri = (modality_tensor >= 9)
    prob_mri = torch.rand(modality_tensor.size(),device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

    # Randomly set a proportion (prob) of the elements to 0
    mask_zero = torch.rand(modality_tensor.size(),device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()
    
    return modality_tensor


def load_filenames(json_data_list, embedding_base_dir, image_dim=None):
    """
    Load filenames and metadata from JSON data list(s) and embedding directory(ies).
    
    Supports both single path (string) and multiple paths (list).

    Args:
        json_data_list (str or list): Path(s) to JSON data list file(s).
        embedding_base_dir (str or list): Base directory(ies) for embeddings.
        image_dim (list or None): Dimension filter. Accepts:
            - None: include all dimensions.
            - [x, y, z]: include only this single dimension.
            - [[x,y,z], [x2,y2,z2], ...]: include any of these dimensions.

    Returns:
        list: List of dicts with keys: 'image' (full path), 'spacing', 'modality', 'dim'.
    """
    # Normalize image_dim to a list-of-dims (or None for "all")
    if image_dim is None:
        allowed_dims = None
    elif isinstance(image_dim[0], int):    # single dim: [128, 128, 128]
        allowed_dims = [image_dim]
    else:                                  # list of dims: [[128,128,128], [256,256,128]]
        allowed_dims = [list(d) for d in image_dim]

    # Normalize to lists
    if isinstance(json_data_list, str):
        json_data_list = [json_data_list]
    if isinstance(embedding_base_dir, str):
        embedding_base_dir = [embedding_base_dir]

    # Ensure matching lengths
    if len(json_data_list) != len(embedding_base_dir):
        if len(embedding_base_dir) == 1:
            # Use single base dir for all JSONs
            embedding_base_dir = embedding_base_dir * len(json_data_list)
        else:
            raise ValueError(
                f"Mismatch: {len(json_data_list)} JSON files but {len(embedding_base_dir)} embedding directories"
            )

    all_files = []
    for json_path, base_dir in zip(json_data_list, embedding_base_dir):
        with open(json_path, "r") as file:
            json_data = json.load(file)

        filenames_train = json_data["training"]
        for _item in filenames_train:
            # Filter by image dimension if specified
            if allowed_dims is not None:
                item_dim = _item.get('dim', None)
                if item_dim is None or item_dim not in allowed_dims:
                    continue
            
            embedding_filename = _item["image"]  # Already includes _emb.nii.gz
            full_path = os.path.join(base_dir, embedding_filename)
            
            # Create file entry with metadata — 'dim' and 'spacing' are required
            for required_field in ('dim', 'spacing'):
                if required_field not in _item:
                    raise ValueError(
                        f"JSON entry is missing required '{required_field}' field: "
                        f"{_item.get('image', '<unknown>')} (in {json_path})"
                    )
            file_entry = {
                'image': full_path,
                'spacing': _item['spacing'],
                'modality': _item.get('class', 'unknown'),  # 'class' field contains modality
                'dim': _item['dim'],
            }
            
            # Include body region indices if available (for anatomical conditioning)
            if 'top_region_index' in _item:
                file_entry['top_region_index'] = _item['top_region_index']
            if 'bottom_region_index' in _item:
                file_entry['bottom_region_index'] = _item['bottom_region_index']
            
            all_files.append(file_entry)
    
    return all_files


def prepare_data(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
    include_modality: bool = True,
    modality_mapping: dict = None,
    use_weighted_sampling: bool = True
) -> tuple:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files with embedded metadata.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        include_body_region (bool): Whether to include body region in data
        include_modality (bool): Whether to include modality information
        modality_mapping (dict): Mapping from modality class names to integers
        use_weighted_sampling (bool): Whether to use weighted random sampling

    Returns:
        tuple: (DataLoader, sampler) Data loader for training and the sampler used.
    """

    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        # Spacing is already in the dict as a list, convert to tensor and scale
        monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x) * 1e2),
    ]
    if include_body_region:
        # Body region indices for anatomical conditioning (head, chest, abdomen, pelvis)
        # These should be included in the train_files dict if body region is used
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: torch.FloatTensor(x)
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    if include_modality:
         train_transforms_list += [ 
             # Modality is already in the dict as a string, map to integer
             monai.transforms.Lambdad(
                keys="modality", func=lambda x: modality_mapping.get(x, 0)
             ),
             monai.transforms.EnsureTyped(keys=['modality'], dtype=torch.long),
         ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    # Create sampler
    if use_weighted_sampling:
        # Extract weights from train_files
        weights = [file_entry.get('sample_weight', 1.0) for file_entry in train_files]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_files),
            replacement=True
        )
        shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, num_workers=6, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
    
    return train_loader, sampler


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def setup_validation(args, device: torch.device, global_rank: int, logger: logging.Logger) -> None:
    """
    Check whether the autoencoder checkpoint exists and set args.validation_enabled.
    Broadcasts the result to all distributed ranks so every process agrees.
    Modifies args.validation_enabled in-place.
    """
    args.validation_enabled = False
    if global_rank == 0:
        autoencoder_path = getattr(args, 'trained_autoencoder_path', None)
        if not autoencoder_path:
            logger.warning("=" * 80)
            logger.warning("⚠️  VALIDATION IMAGE GENERATION DISABLED ⚠️")
            logger.warning("=" * 80)
            logger.warning("Reason: Autoencoder path not configured in config file")
            logger.warning("Validation image generation requires the autoencoder model.")
            logger.warning("Please set 'trained_autoencoder_path' in your config.")
            logger.warning("Validation will be skipped for all epochs.")
            logger.warning("=" * 80)
        elif not os.path.exists(autoencoder_path):
            logger.warning("=" * 80)
            logger.warning("⚠️  VALIDATION IMAGE GENERATION DISABLED ⚠️")
            logger.warning("=" * 80)
            logger.warning(f"Reason: Autoencoder file not found: {autoencoder_path}")
            logger.warning("Validation image generation requires the autoencoder model.")
            logger.warning("Please ensure the autoencoder checkpoint exists before running validation.")
            logger.warning("Validation will be skipped for all epochs.")
            logger.warning("=" * 80)
        else:
            args.validation_enabled = True
            logger.info(f"Autoencoder file found: {autoencoder_path} (validation will run every {VAL_INTERVAL} epochs)")
    if dist.is_initialized():
        validation_enabled_tensor = torch.tensor([1 if args.validation_enabled else 0], dtype=torch.int, device=device)
        dist.broadcast(validation_enabled_tensor, src=0)
        args.validation_enabled = bool(validation_enabled_tensor.item())


def filter_existing_files(train_files: list, logger: logging.Logger) -> list:
    """
    Return only the entries from train_files whose image path exists on disk.
    Missing files are logged as warnings (RankFilter ensures only rank 0 prints).
    """
    valid = []
    for file_entry in train_files:
        if not os.path.exists(file_entry['image']):
            logger.warning(f"File not found, skipping: {file_entry['image']}")
        else:
            valid.append(file_entry)
    logger.info(f"Valid training files: {len(valid)} / {len(train_files)}")
    return valid


def compute_modality_weights(
    valid_train_files: list,
    max_weight: float = 10.0,
    logger: logging.Logger = None,
) -> tuple:
    """
    Compute inverse-frequency sampling weights per modality.
    Weights are normalized so the most-common modality gets weight 1.0, then
    capped at max_weight.  Each file entry in valid_train_files is modified
    in-place by adding a 'sample_weight' key.

    Args:
        valid_train_files: List of file entry dicts (must have 'modality' key).
        max_weight: Upper cap on any single modality's weight.
        logger: If provided, logs the modality distribution.

    Returns:
        (modality_counts, modality_weights)
    """
    modality_counts: dict = {}
    for file_entry in valid_train_files:
        mod = file_entry['modality']
        modality_counts[mod] = modality_counts.get(mod, 0) + 1

    total_files = len(valid_train_files)
    # Raw inverse-frequency weights
    raw_weights = {mod: total_files / count for mod, count in modality_counts.items()}

    # Normalize so the most-common modality (smallest raw weight) becomes 1.0
    min_raw = min(raw_weights.values())
    weight_norm_factor = 1.0 / min_raw

    # Apply normalization + cap
    modality_weights = {
        mod: min(w * weight_norm_factor, max_weight)
        for mod, w in raw_weights.items()
    }

    # Assign to each file entry in-place
    for file_entry in valid_train_files:
        file_entry['sample_weight'] = modality_weights[file_entry['modality']]

    if logger is not None:
        logger.info(f"Modality distribution (max_weight={max_weight}):")
        for mod, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
            normalized_weight = raw_weights[mod] * weight_norm_factor
            capped_weight = modality_weights[mod]
            if normalized_weight > max_weight:
                logger.info(
                    f"  {mod}: {count} files (weight: {capped_weight:.4f}, "
                    f"capped from {normalized_weight:.4f})"
                )
            else:
                logger.info(f"  {mod}: {count} files (weight: {capped_weight:.4f})")

    return modality_counts, modality_weights


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
        device: torch.device,
        logger: logging.Logger,
        global_rank: int,
        amp: bool = True,
        max_iter: int = None,
    ) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        global_rank (int): Global rank for distributed training (unique across all nodes).
        amp (bool): Use automatic mixed precision training.
        max_iter (int): Maximum iterations per epoch. If None, process all data.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    # Access model attributes through .module if wrapped in DDP
    model_for_attr = unet.module if isinstance(unet, DistributedDataParallel) else unet
    include_body_region = model_for_attr.include_top_region_index_input
    include_modality = model_for_attr.num_class_embeds is not None

    # RankFilter already ensures only rank 0 logs
    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    for train_data in train_loader:
        if max_iter is not None and _iter >= max_iter:
            break
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = train_data["modality"].to(device)     
            modality_tensor = augment_modality_label(modality_tensor).to(device)

        spacing_tensor = train_data["spacing"].to(device)
        # Add random scaling to spacing for data augmentation with prob=0.3
        # Independent scale factors for each axis (x, y, z), sampled uniformly from [0.95, 1.05] (5% variation)
        if torch.rand(1, device=device) < 0.5:
            spacing_scale = torch.rand_like(spacing_tensor) * 0.1 + 0.95  # [0.95, 1.05] for each axis
            spacing_tensor = spacing_tensor * spacing_scale

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            loss = loss_pt(model_output.float(), model_gt.float())
            
            # Check for NaN loss and terminate if detected
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("=" * 80)
                logger.error("❌ TRAINING TERMINATED: NaN or Inf loss detected!")
                logger.error("=" * 80)
                logger.error(f"Epoch: {epoch + 1}, Iteration: {_iter}")
                logger.error(f"Loss value: {loss.item()}")
                logger.error("Training cannot continue. Please check:")
                logger.error("  - Learning rate (may be too high)")
                logger.error("  - Model architecture")
                logger.error("  - Data quality")
                logger.error("  - Gradient clipping")
                logger.error("=" * 80)
                if dist.is_initialized():
                    dist.destroy_process_group()
                sys.exit(1)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        # RankFilter already ensures only rank 0 logs
        if _iter % 10 == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def generate_validation_images_for_modalities(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    unet: torch.nn.Module,
    scale_factor: torch.Tensor,
    device: torch.device,
) -> list:
    """
    Generate validation images for all modalities specified in config.
    
    Args:
        env_config_path (str): Path to environment config.
        model_config_path (str): Path to model config.
        model_def_path (str): Path to model definition.
        args (argparse.Namespace): Configuration arguments.
        logger (logging.Logger): Logger for logging information.
    
    Returns:
        list: List of tuples (modality_id, modality_name, image_data_3d)
              where image_data_3d is a numpy array of shape [D, H, W]
    """
    import tempfile
    import nibabel as nib
    
    generated_images = []
    
    try:
        # Load modality mapping for reverse lookup
        modality_id_to_name = {}
        if hasattr(args, 'modality_mapping_path') and args.modality_mapping_path:
            try:
                with open(args.modality_mapping_path, 'r') as f:
                    modality_name_to_id = json.load(f)
                    # Reverse mapping: id -> name
                    modality_id_to_name = {v: k for k, v in modality_name_to_id.items()}
            except Exception as e:
                logger.warning(f"Could not load modality mapping: {e}")
        
        # Load the inference config to check if modality is a list or scalar
        with open(model_config_path, 'r') as f:
            if model_config_path.endswith('.yaml') or model_config_path.endswith('.yml'):
                model_config = yaml.safe_load(f)
            else:
                model_config = json.load(f)
        
        if not model_config or 'diffusion_unet_inference' not in model_config:
            logger.warning("Could not parse model config for validation")
            return generated_images
        
        modality_config = model_config['diffusion_unet_inference'].get("modality", 9)
        
        # Determine if modality is a list or scalar
        if isinstance(modality_config, list):
            modality_list = modality_config
            logger.info(f"Generating one image per modality: {modality_list}")
        else:
            modality_list = [modality_config]
            logger.info(f"Generating image for single modality: {modality_config}")
        
        # Load inference config to get inference parameters
        inference_args = load_config(env_config_path, model_config_path, model_def_path)
        
        # Load autoencoder (UNet and scale_factor already available from training)
        autoencoder = define_instance(inference_args, "autoencoder_def").to(device)
        if not os.path.exists(inference_args.trained_autoencoder_path):
            logger.warning(f"Autoencoder checkpoint not found: {inference_args.trained_autoencoder_path}")
            return generated_images
        
        try:
            checkpoint_autoencoder = torch.load(inference_args.trained_autoencoder_path, map_location=device, weights_only=False)
            if "unet_state_dict" in checkpoint_autoencoder.keys():
                checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
            autoencoder.load_state_dict(checkpoint_autoencoder)
            logger.info(f"Loaded autoencoder from {inference_args.trained_autoencoder_path}")
        except Exception as e:
            logger.warning(f"Failed to load autoencoder: {e}")
            return generated_images
        
        # Get inference parameters
        output_size = tuple(inference_args.diffusion_unet_inference["dim"])
        out_spacing = tuple(inference_args.diffusion_unet_inference["spacing"])
        inference_args.cfg_guidance_scale = inference_args.diffusion_unet_inference["cfg_guidance_scale"]
        
        # Calculate divisor for downsample level
        num_downsample_level = max(
            1,
            (
                len(inference_args.diffusion_unet_def["num_channels"])
                if isinstance(inference_args.diffusion_unet_def["num_channels"], list)
                else len(inference_args.diffusion_unet_def["attention_levels"])
            ),
        )
        divisor = 2 ** (num_downsample_level - 2)

        # Free cached training memory before running inference
        torch.cuda.empty_cache()
        logger.info("[validation] Cleared CUDA cache before inference.")

        unet.eval()
        try:
            # Generate one image per modality
            for modality_id in modality_list:
                # Get modality name for labeling
                modality_name = modality_id_to_name.get(modality_id, f"mod_{modality_id}")
                
                # Update modality in inference args
                inference_args.diffusion_unet_inference["modality"] = modality_id
                
                try:
                    # Prepare tensors for this modality
                    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(
                        inference_args, device
                    )
                    
                    # Run inference directly (no distributed setup/teardown)
                    syn_data = run_inference(
                        inference_args,
                        device,
                        autoencoder,
                        unet,
                        scale_factor,
                        top_region_index_tensor,
                        bottom_region_index_tensor,
                        spacing_tensor,
                        modality_tensor,
                        output_size,
                        divisor,
                        logger,
                    )
                    
                    # Store the generated image data
                    generated_images.append((modality_id, modality_name, syn_data))
                    logger.info(f"Generated validation image for {modality_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to generate image for {modality_name}: {str(e)}")
                    import traceback
                    logger.warning(traceback.format_exc())
        finally:
            unet.train()
            torch.cuda.empty_cache()
    
    except Exception as e:
        logger.warning(f"Validation image generation failed: {str(e)}")
        import traceback
        logger.warning(traceback.format_exc())
    
    return generated_images


def log_validation_images_to_tensorboard(
    generated_images: list,
    epoch: int,
    tensorboard_writer: SummaryWriter,
    logger: logging.Logger,
) -> None:
    """
    Log generated validation images to TensorBoard.
    
    Args:
        generated_images (list): List of tuples (modality_id, modality_name, image_data_3d)
        epoch (int): Current epoch number.
        tensorboard_writer (SummaryWriter): TensorBoard writer.
        logger (logging.Logger): Logger for logging information.
    """
    from .utils_plot import find_label_center_loc, get_xyz_plot
    
    for modality_id, modality_name, syn_data in generated_images:
        try:
            # Convert to tensor [1, D, H, W]
            image_volume = torch.from_numpy(syn_data).unsqueeze(0)
            
            # Clip and normalize to [0, 1]
            # For MR images, clip to [0, 1000] as in the tutorial
            image_volume = torch.clip(image_volume, 0, 1000)
            image_volume = image_volume - torch.min(image_volume)
            image_volume = image_volume / (torch.max(image_volume) + 1e-8)
            
            # Find center voxel location for 2D slice visualization
            # Using find_label_center_loc as in the tutorial
            center_loc_axis = find_label_center_loc(torch.flip(image_volume[0, ...], [-3, -2, -1]))
            
            # Convert metatensor to regular list if needed
            center_loc_axis = [int(c) if c is not None else image_volume.shape[i+1] // 2 
                              for i, c in enumerate(center_loc_axis)]
            
            # Use get_xyz_plot to create concatenated visualization (as in tutorial)
            # mask_bool=False for grayscale images (not label masks)
            vis_image = get_xyz_plot(
                image=image_volume,
                center_loc_axis=center_loc_axis,
                mask_bool=False,  # Not a mask, just grayscale image
            )
            
            # Convert from [H, W, 3] numpy array to [3, H, W] tensor for TensorBoard
            # vis_image is already in float32 [0, 1] when mask_bool=False
            vis_image_tensor = torch.from_numpy(vis_image).permute(2, 0, 1).float()
            
            # Log the combined XYZ view
            tensorboard_writer.add_image(f'val/{modality_name}_xyz', vis_image_tensor, epoch + 1)
            
            logger.info(f"Logged validation image for {modality_name} (XYZ view) to TensorBoard")
        
        except Exception as e:
            logger.warning(f"Failed to log {modality_name} to TensorBoard: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
    
    # Flush TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.flush()


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, amp: bool = True
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        amp (bool): Use automatic mixed precision training.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, global_rank, world_size, device = initialize_distributed()
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")
    
    # Check autoencoder availability and broadcast validation_enabled to all ranks
    setup_validation(args, device, global_rank, logger)
    
    # Initialize TensorBoard writer on main process only
    tensorboard_writer = None
    if global_rank == 0:
        tensorboard_path = args.tfevent_dir
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        logger.info(f"TensorBoard logging to: {tensorboard_path}")
    
    # Read image_dim from config file
    image_dim = args.diffusion_unet_train.get('image_dim', None)
    # Convert to list if it's a different type (e.g., tuple from YAML)
    if image_dim is not None and not isinstance(image_dim, list):
        image_dim = list(image_dim)

    # RankFilter already ensures only rank 0 logs
    logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
    logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
    logger.info(f"[config] data_list -> {args.json_data_list}.")
    logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
    logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
    logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")
    if image_dim is not None:
        logger.info(f"[config] image_dim filter -> {image_dim}.")
    else:
        logger.info(f"[config] image_dim filter -> None (using all dimensions).")

    # File operation - must be on rank 0 only
    if global_rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet, scale_factor, start_epoch, optimizer_state_dict, scheduler_state_dict = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    
    # Access model attributes through .module if wrapped in DDP
    model_for_attr = unet.module if isinstance(unet, DistributedDataParallel) else unet
    include_body_region = model_for_attr.include_top_region_index_input
    include_modality = (model_for_attr.num_class_embeds is not None)
    
    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    # Load filenames and metadata from JSON(s) - now supports both single and multiple
    train_files = load_filenames(args.json_data_list, args.embedding_base_dir, image_dim=image_dim)
    # RankFilter already ensures only rank 0 logs
    logger.info(f"num_files_train: {len(train_files)}")
    if image_dim is not None:
        logger.info(f"Filtered by image_dim: {image_dim}")

    # Filter out non-existent files, then compute modality-based sampling weights
    valid_train_files = filter_existing_files(train_files, logger)
    compute_modality_weights(valid_train_files, max_weight=10.0, logger=logger)
    
    if dist.is_initialized():
        valid_train_files = partition_dataset(
            data=valid_train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[global_rank]

    # Get use_weighted_sampling from config (default to True if not specified)
    use_weighted_sampling = args.diffusion_unet_train.get("use_weighted_sampling", True)
    # RankFilter already ensures only rank 0 logs
    logger.info(f"Weighted sampling: {'enabled' if use_weighted_sampling else 'disabled'}")

    train_loader, sampler = prepare_data(
        valid_train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping = args.modality_mapping,
        use_weighted_sampling=use_weighted_sampling
    )

    # Calculate or load scale factor
    if scale_factor is None:
        scale_factor = calculate_scale_factor(train_loader, device, logger)
    else:
        logger.info(f"Using loaded scale_factor: {scale_factor}.")
    
    # Create optimizer and load state if available (only when resuming, not training from scratch)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    if start_epoch > 0 and optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info("Loaded optimizer state from checkpoint.")
    elif optimizer_state_dict is not None:
        logger.warning("Skipping optimizer state loading.")

    # Create learning rate scheduler
    total_steps = args.diffusion_unet_train["n_epochs"] - start_epoch
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    if start_epoch > 0 and scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(scheduler_state_dict)
        logger.info("Loaded scheduler state from checkpoint.")
    elif scheduler_state_dict is not None:
        logger.warning("Skipping lr_scheduler state loading.")
    
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(start_epoch, args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            global_rank,
            amp=amp,
        )
        lr_scheduler.step()  # Step LR once per epoch

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or global_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            
            # Log training loss to TensorBoard
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('train/loss', loss_torch_epoch, epoch + 1)

            # Save latest checkpoint every epoch (format auto-detected from filename)
            save_checkpoint(
                epoch,
                unet,
                loss_torch_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args.model_filename,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch_finished=True,
            )
            
            # Save periodic checkpoint every SAVE_EPOCH_INTERVAL epochs
            if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
                # Preserve the file extension when creating periodic checkpoint filename
                if args.model_filename.endswith('.safetensors'):
                    epoch_filename = args.model_filename.replace('.safetensors', f'_epoch{epoch + 1}.safetensors')
                else:
                    epoch_filename = args.model_filename.replace('.pt', f'_epoch{epoch + 1}.pt')
                
                save_checkpoint(
                    epoch,
                    unet,
                    loss_torch_epoch,
                    args.noise_scheduler["num_train_timesteps"],
                    scale_factor,
                    args.model_dir,
                    epoch_filename,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch_finished=True,
                )
                logger.info(f"Saved periodic checkpoint: {epoch_filename}")
            
            # Generate and log validation images every VAL_INTERVAL epochs (only if validation is enabled)
            if args.validation_enabled and (epoch + 1) % VAL_INTERVAL == 0:
                logger.info(f"Generating validation images at epoch {epoch + 1}...")
                
                # Generate images for all modalities
                generated_images = generate_validation_images_for_modalities(
                    env_config_path,
                    model_config_path,
                    model_def_path,
                    args,
                    logger,
                    unet,
                    scale_factor,
                    device,
                )
                
                # Log to TensorBoard
                if generated_images and tensorboard_writer is not None:
                    log_validation_images_to_tensorboard(
                        generated_images,
                        epoch,
                        tensorboard_writer,
                        logger,
                    )
    
    # Close TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
        logger.info("TensorBoard writer closed.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "-e",
        "--env_config_path",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--model_config_path",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "-t",
        "--model_def_path", 
        type=str, 
        default="./configs/config_maisi.json", 
        help="Path to model definition file"
    )
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")

    args = parser.parse_args()
    diff_model_train(args.env_config_path, args.model_config_path, args.model_def_path, args.amp)
