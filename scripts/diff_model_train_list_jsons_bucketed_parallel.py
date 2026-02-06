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
Diffusion Model Training with Bucketed Data Parallel Training

Key Difference vs diff_model_train_list_jsons.py:
  - This script: Distributes different image dimensions across GPUs (bucketed data parallel training)
    * Each GPU processes specific image sizes (e.g., GPU0: 128^3, GPU1: 512^3)
    * Batch size is auto-adjusted per GPU based on image size
    * Limits to MAX_ITER iterations/epoch for balanced training to avoid error
  
  - Base script: Standard even partitioning, processes all data each epoch

When to Use:
  - This script: Mixed image sizes, want memory optimization
  - Base script: Single image size or all dimensions with even partitioning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import math
import builtins
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

from diff_model_setting import initialize_distributed, load_config, setup_logging
from utils import define_instance, load_checkpoint, save_checkpoint

# Import reusable functions from diff_model_train_list_jsons
from diff_model_train_list_jsons import (
    augment_modality_label,
    load_unet,
    load_filenames,
    prepare_data,
    calculate_scale_factor,
    create_optimizer,
    create_lr_scheduler,
    train_one_epoch,
    generate_validation_images_for_modalities,
    log_validation_images_to_tensorboard,
    SAVE_EPOCH_INTERVAL,
)

# Max iterations per epoch for by_size bucketing
MAX_ITER = 50  # Balanced training across image sizes

# Reference image size for batch size calculation (256x256x128)
REFERENCE_SIZE_PRODUCT = 256 * 256 * 128


def analyze_data_distribution(
    all_train_files: list,
    base_batch_size: int,
    logger: logging.Logger,
) -> dict:
    """
    Analyze data distribution and calculate batch counts per dimension.
    
    Args:
        all_train_files (list): All training files with 'dim' metadata.
        base_batch_size (int): Base batch size (for reference size 256x256x128).
        logger (logging.Logger): Logger.
    
    Returns:
        dict: {dimension_str: {'count': num_files, 'batch_size': batch_size, 'num_batches': num_batches}}
    """
    # Count files per dimension
    dim_counts = defaultdict(int)
    for f in all_train_files:
        dim = f.get('dim')
        if dim is None:
            raise ValueError(
                f"File entry missing 'dim' field: {f.get('image', 'unknown')}. "
                f"Bucketed parallel training requires all files to have dimension metadata."
            )
        dim_str = 'x'.join(map(str, dim))
        dim_counts[dim_str] += 1
    
    # Calculate batch size and number of batches per dimension
    dim_info = {}
    
    for dim_str, count in dim_counts.items():
        dimensions = list(map(int, dim_str.split('x')))
        size_product = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Calculate batch size (same formula as before)
        batch_size = int(base_batch_size * REFERENCE_SIZE_PRODUCT / size_product)
        batch_size = max(batch_size, 1)
        batch_size = min(batch_size, 32)
        
        num_batches = math.ceil(float(count) / batch_size)
        
        dim_info[dim_str] = {
            'count': count,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'size_product': size_product,
        }
    
    # Log distribution
    logger.info(f"[bucketed_parallel] Data distribution analysis:")
    for dim_str in sorted(dim_info.keys(), key=lambda x: dim_info[x]['size_product']):
        info = dim_info[dim_str]
        logger.info(f"  {dim_str}: {info['count']} files, batch_size={info['batch_size']}, "
                   f"num_batches={info['num_batches']}")
    
    return dim_info


def distribute_dimensions_across_gpus(
    dim_info: dict,
    world_size: int,
    logger: logging.Logger,
) -> list:
    """
    Distribute dimensions across GPUs to balance number of batches per GPU.
    
    If num_dimensions > num_gpus: Select top dimensions by sample count.
    If num_dimensions <= num_gpus: Use greedy algorithm to balance batches.
    
    Args:
        dim_info (dict): Dimension info from analyze_data_distribution.
        world_size (int): Total number of GPUs.
        logger (logging.Logger): Logger.
    
    Returns:
        list: Size list for each GPU (length = world_size)
    """
    num_dimensions = len(dim_info)
    
    if num_dimensions > world_size:
        # More dimensions than GPUs: Pick top dimensions by num_batches
        logger.info(f"[bucketed_parallel] {num_dimensions} dimensions > {world_size} GPUs")
        logger.info(f"[bucketed_parallel] Selecting top {world_size} dimensions by num_batches")
        
        # Sort by num_batches (number of batches) descending
        sorted_by_batches = sorted(dim_info.items(), key=lambda x: -x[1]['num_batches'])
        selected_dims = [dim_str for dim_str, _ in sorted_by_batches[:world_size]]
        
        # Create size_list (one dimension per GPU)
        size_list = selected_dims
        
        # Log what was selected and what was skipped
        logger.info(f"[bucketed_parallel] Selected dimensions:")
        for i, dim_str in enumerate(selected_dims):
            info = dim_info[dim_str]
            logger.info(f"  GPU {i}: {dim_str} - {info['count']} files, {info['num_batches']} batches")
        
        skipped_dims = [dim_str for dim_str, _ in sorted_by_batches[world_size:]]
        if skipped_dims:
            logger.warning(f"[bucketed_parallel] Skipped dimensions (too few GPUs):")
            for dim_str in skipped_dims:
                info = dim_info[dim_str]
                logger.warning(f"  {dim_str}: {info['count']} files, {info['num_batches']} batches")
        
        # Calculate balance metrics
        selected_batches = [dim_info[dim_str]['num_batches'] for dim_str in selected_dims]
        avg_batches = sum(selected_batches) / len(selected_batches)
        max_imbalance = max(abs(b - avg_batches) for b in selected_batches)
        logger.info(f"[bucketed_parallel] Balance: avg={avg_batches:.1f} batches/GPU, max_imbalance={max_imbalance:.1f}")
        
    else:
        # Fewer or equal dimensions than GPUs: Use greedy balancing
        logger.info(f"[bucketed_parallel] {num_dimensions} dimensions <= {world_size} GPUs")
        logger.info(f"[bucketed_parallel] Using greedy algorithm to balance batches")
        
        # Initialize batch count per GPU
        gpu_batches = [0] * world_size
        gpu_assignments = [[] for _ in range(world_size)]
        
        # Sort dimensions by number of batches (descending) for better balance
        sorted_dims = sorted(dim_info.items(), key=lambda x: -x[1]['num_batches'])
        
        # Greedy assignment: assign each dimension to GPU with fewest batches
        for dim_str, info in sorted_dims:
            # Find GPU with minimum batches
            min_gpu = min(range(world_size), key=lambda i: gpu_batches[i])
            
            # Assign this dimension to that GPU
            gpu_assignments[min_gpu].append(dim_str)
            gpu_batches[min_gpu] += info['num_batches']
        
        # Create size_list (one entry per GPU, may have empty GPUs)
        size_list = []
        for gpu_id, gpu_dims in enumerate(gpu_assignments):
            if gpu_dims:
                # Use the first (largest) dimension assigned to this GPU
                size_list.append(gpu_dims[0])
            else:
                # Empty GPU: assign smallest dimension as fallback
                size_list.append(min(dim_info.keys(), key=lambda x: dim_info[x]['size_product']))
        
        # Log distribution
        logger.info(f"[bucketed_parallel] GPU dimension assignment (balanced by batches):")
        for gpu_id in range(world_size):
            if gpu_id < len(gpu_assignments):
                dims = gpu_assignments[gpu_id]
                batches = gpu_batches[gpu_id]
                logger.info(f"  GPU {gpu_id}: {dims} ({batches} batches)")
        
        total_batches = sum(gpu_batches)
        avg_batches = total_batches / world_size if world_size > 0 else 0
        max_imbalance = max(abs(b - avg_batches) for b in gpu_batches)
        logger.info(f"[bucketed_parallel] Balance: avg={avg_batches:.1f} batches/GPU, max_imbalance={max_imbalance:.1f}")
    
    logger.info(f"[bucketed_parallel] Total unique dimensions: {len(dim_info)}, GPUs: {world_size}, "
               f"size_list length: {len(size_list)}")
    
    return size_list


def partition_data_by_size(
    all_train_files: list,
    global_rank: int,
    world_size: int,
    logger: logging.Logger,
    base_batch_size: int = 1,
) -> tuple:
    """
    Partition data by image size, assigning specific dimensions to each GPU with dynamic batch sizing.
    
    This function dynamically analyzes the data distribution and assigns dimensions to GPUs
    to balance the number of batches per GPU.
    
    Args:
        all_train_files (list): All training files with 'dim' metadata.
        global_rank (int): Current GPU's global rank.
        world_size (int): Total number of GPUs.
        logger (logging.Logger): Logger.
        base_batch_size (int): Base batch size (for reference size 256x256x128).
    
    Returns:
        tuple: (train_files_for_rank, batch_size_for_rank, dimensions_assigned)
    """
    # Analyze data distribution (only rank 0 logs details)
    if global_rank == 0:
        dim_info = analyze_data_distribution(all_train_files, base_batch_size, logger)
        size_list = distribute_dimensions_across_gpus(dim_info, world_size, logger)
    else:
        # Other ranks need to compute the same distribution
        dim_counts = defaultdict(int)
        for f in all_train_files:
            dim = f.get('dim')
            if dim is None:
                raise ValueError(
                    f"File entry missing 'dim' field: {f.get('image', 'unknown')}. "
                    f"Bucketed parallel training requires all files to have dimension metadata."
                )
            dim_str = 'x'.join(map(str, dim))
            dim_counts[dim_str] += 1
        
        dim_info = {}
        for dim_str, count in dim_counts.items():
            dimensions = list(map(int, dim_str.split('x')))
            size_product = dimensions[0] * dimensions[1] * dimensions[2]
            batch_size = max(1, min(32, int(base_batch_size * REFERENCE_SIZE_PRODUCT / size_product)))
            num_batches = math.ceil(float(count) / batch_size)
            dim_info[dim_str] = {'count': count, 'batch_size': batch_size, 'num_batches': num_batches, 'size_product': size_product}
        
        num_dimensions = len(dim_info)
        
        if num_dimensions > world_size:
            # More dimensions than GPUs: Pick top dimensions by num_batches
            sorted_by_batches = sorted(dim_info.items(), key=lambda x: -x[1]['num_batches'])
            size_list = [dim_str for dim_str, _ in sorted_by_batches[:world_size]]
        else:
            # Fewer or equal dimensions: Greedy balancing
            gpu_batches = [0] * world_size
            gpu_assignments = [[] for _ in range(world_size)]
            sorted_dims = sorted(dim_info.items(), key=lambda x: -x[1]['num_batches'])
            for dim_str, info in sorted_dims:
                min_gpu = min(range(world_size), key=lambda i: gpu_batches[i])
                gpu_assignments[min_gpu].append(dim_str)
                gpu_batches[min_gpu] += info['num_batches']
            
            size_list = []
            for gpu_id, gpu_dims in enumerate(gpu_assignments):
                if gpu_dims:
                    size_list.append(gpu_dims[0])
                else:
                    size_list.append(min(dim_info.keys(), key=lambda x: dim_info[x]['size_product']))
    
    size_list.sort()
    
    if global_rank == 0:
        logger.info(f"[bucketed_parallel] Size distribution across {world_size} GPUs: {set(size_list)}")
        logger.info(f"[bucketed_parallel] Size counts: {dict((s, size_list.count(s)) for s in set(size_list))}")
    
    size_rank = global_rank % len(size_list)
    list_builtin = builtins.list
    size_str = size_list[size_rank]
    dimensions = list_builtin(map(int, size_str.split('x')))
    size_product = dimensions[0] * dimensions[1] * dimensions[2]
    
    logger.info(f"[bucketed_parallel] Rank {global_rank}: assigned dimension={dimensions}")
    
    train_files_filtered = [f for f in all_train_files if f.get('dim') == dimensions]
    
    logger.info(f"[bucketed_parallel] Rank {global_rank}: found {len(train_files_filtered)} files with dim={dimensions}")
    
    size_str_count = size_list.count(size_str)
    if size_str_count > 1:
        train_files_filtered = partition_dataset(
            data=train_files_filtered,
            shuffle=True,
            num_partitions=size_str_count,
            even_divisible=True
        )[size_rank % size_str_count]
        logger.info(f"[bucketed_parallel] Rank {global_rank}: partitioned among {size_str_count} ranks, "
                   f"subset {size_rank % size_str_count}, final count={len(train_files_filtered)}")
    
    batch_size = int(base_batch_size * REFERENCE_SIZE_PRODUCT / size_product)
    batch_size = max(batch_size, 1)
    batch_size = min(batch_size, 32)
    
    logger.info(f"[bucketed_parallel] Rank {global_rank}: calculated batch_size={batch_size} for size {size_str}")
    
    num_batch = float(len(train_files_filtered)) / batch_size
    if math.ceil(num_batch) < MAX_ITER:
        replication_factor = math.ceil(float(MAX_ITER) / num_batch)
        train_files_filtered = train_files_filtered * replication_factor
        logger.info(f"[bucketed_parallel] Rank {global_rank}: replicated data {replication_factor}x "
                   f"to ensure {MAX_ITER} iterations ({len(train_files_filtered)} total files)")
    
    return train_files_filtered, batch_size, dimensions


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, amp: bool = True
) -> None:
    """Main training function with by-size bucketing support."""
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed()
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")
    
    tensorboard_writer = None
    if local_rank == 0:
        tensorboard_path = args.tfevent_dir
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        logger.info(f"TensorBoard logging to: {tensorboard_path}")
    
    # This script always uses by_size bucketing (partition_type is implicit)
    logger.info(f"[bucketed_parallel] Using by_size bucketing (script choice determines partition type)")
    logger.info(f"[bucketed_parallel] MAX_ITER set to {MAX_ITER} iterations per epoch")
    
    image_dim = args.diffusion_unet_train.get('image_dim', None)
    if image_dim is not None:
        logger.warning(f"[bucketed_parallel] image_dim filter ({image_dim}) is ignored - all dimensions loaded for bucketing")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet, scale_factor, start_epoch, optimizer_state_dict, scheduler_state_dict = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    
    model_for_attr = unet.module if isinstance(unet, DistributedDataParallel) else unet
    include_body_region = model_for_attr.include_top_region_index_input
    include_modality = (model_for_attr.num_class_embeds is not None)
    
    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    train_files = load_filenames(
        args.json_data_list,
        args.embedding_base_dir,
        image_dim=None  # Load all dimensions for by_size bucketing
    )
    if local_rank == 0:
        logger.info(f"num_files_train (all): {len(train_files)}")

    valid_train_files = []
    for file_entry in train_files:
        if not os.path.exists(file_entry['image']):
            if local_rank == 0:
                logger.warning(f"File not found, skipping: {file_entry['image']}")
            continue
        valid_train_files.append(file_entry)
    
    if local_rank == 0:
        logger.info(f"Valid training files: {len(valid_train_files)}")
    
    modality_counts = {}
    for file_entry in valid_train_files:
        modality = file_entry['modality']
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    total_files = len(valid_train_files)
    max_weight = 100.0
    modality_weights = {}
    for mod, count in modality_counts.items():
        weight = total_files / count
        modality_weights[mod] = min(weight, max_weight)
    
    for file_entry in valid_train_files:
        file_entry['sample_weight'] = modality_weights[file_entry['modality']]
    
    if local_rank == 0:
        logger.info(f"Modality distribution (max_weight={max_weight}):")
        for mod, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
            raw_weight = total_files / count
            capped_weight = modality_weights[mod]
            if raw_weight > max_weight:
                logger.info(f"  {mod}: {count} files (weight: {capped_weight:.4f}, capped from {raw_weight:.4f})")
            else:
                logger.info(f"  {mod}: {count} files (weight: {capped_weight:.4f})")
    
    # By-size bucketing: partition data by image dimensions
    partitioned_train_files, batch_size, assigned_dims = partition_data_by_size(
        all_train_files=valid_train_files,
        global_rank=dist.get_rank() if dist.is_initialized() else 0,
        world_size=world_size,
        logger=logger,
        base_batch_size=args.diffusion_unet_train["batch_size"],
    )
    logger.info(f"[bucketed_parallel] Rank {local_rank}: {len(partitioned_train_files)} files, "
               f"batch_size={batch_size}, dims={assigned_dims}")

    use_weighted_sampling = args.diffusion_unet_train.get("use_weighted_sampling", True)
    if local_rank == 0:
        logger.info(f"Weighted sampling: {'enabled' if use_weighted_sampling else 'disabled'}")

    train_loader, sampler = prepare_data(
        partitioned_train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=batch_size,
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping = args.modality_mapping,
        use_weighted_sampling=use_weighted_sampling
    )

    if scale_factor is None:
        scale_factor = calculate_scale_factor(train_loader, device, logger)
    else:
        logger.info(f"Using loaded scale_factor: {scale_factor}.")
    
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info("Loaded optimizer state from checkpoint.")

    # Calculate total steps for by_size bucketing
    total_steps = (args.diffusion_unet_train["n_epochs"] - start_epoch) * MAX_ITER
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    if scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(scheduler_state_dict)
        logger.info("Loaded scheduler state from checkpoint.")
    
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
            batch_size,
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
            max_iter=MAX_ITER,  # Always use MAX_ITER for by_size bucketing
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('train/loss', loss_torch_epoch, epoch + 1)

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
            
            if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
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
            
            if VALIDATION_AVAILABLE and (epoch + 1) % 50 == 0:
                logger.info(f"Generating validation images at epoch {epoch + 1}...")
                
                generated_images = generate_validation_images_for_modalities(
                    env_config_path,
                    model_config_path,
                    model_def_path,
                    args,
                    logger,
                )
                
                if generated_images and tensorboard_writer is not None:
                    log_validation_images_to_tensorboard(
                        generated_images,
                        epoch,
                        tensorboard_writer,
                        logger,
                    )
    
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
