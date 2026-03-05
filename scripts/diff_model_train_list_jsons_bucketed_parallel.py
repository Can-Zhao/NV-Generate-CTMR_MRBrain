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

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance, load_checkpoint, load_unet, save_checkpoint

# Import reusable functions from diff_model_train_list_jsons
from .diff_model_train_list_jsons import (
    load_filenames,
    prepare_data,
    calculate_scale_factor,
    setup_validation,
    filter_existing_files,
    compute_modality_weights,
    create_optimizer,
    create_lr_scheduler,
    train_one_epoch,
    generate_validation_images_for_modalities,
    log_validation_images_to_tensorboard,
    SAVE_EPOCH_INTERVAL,
    VAL_INTERVAL,
)

# Reference image size for batch size calculation (128x128x128)
REFERENCE_SIZE_PRODUCT = 128 * 128 * 128


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
        batch_size = min(batch_size, 96)
        
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


def _compute_size_list(dim_info: dict, world_size: int) -> list:
    """
    Pure allocation logic: compute size_list (one entry per GPU) from dim_info.

    Strategy:
      - If num_dimensions > world_size: select the top `world_size` dimensions by
        num_batches (one GPU each).
      - If num_dimensions <= world_size: proportionally allocate GPUs to dimensions
        using a greedy "most-loaded first" algorithm.  Each dimension starts with 1
        GPU; the remaining GPUs are awarded one-by-one to whichever dimension currently
        has the highest batches-per-GPU ratio.  This distributes ALL GPUs and achieves
        near-optimal load balance.

    Args:
        dim_info (dict): {dim_str: {'num_batches': int, 'size_product': int, ...}}
        world_size (int): Total number of GPUs.

    Returns:
        list: Sorted list of dim strings, length == world_size.
              Dimensions that receive multiple GPUs appear multiple times.
    """
    num_dimensions = len(dim_info)

    if num_dimensions > world_size:
        # One GPU per dimension, pick the busiest world_size dims
        sorted_by_batches = sorted(dim_info.items(), key=lambda x: -x[1]['num_batches'])
        size_list = [dim_str for dim_str, _ in sorted_by_batches[:world_size]]
    else:
        # Proportional allocation: every GPU is assigned a real dimension
        # Start each dimension with 1 GPU
        gpu_counts = {dim_str: 1 for dim_str in dim_info}
        remaining = world_size - num_dimensions

        # Greedily add GPUs to the most-loaded dimension (highest batches/GPU)
        while remaining > 0:
            most_loaded = max(
                dim_info.keys(),
                key=lambda d: dim_info[d]['num_batches'] / gpu_counts[d],
            )
            gpu_counts[most_loaded] += 1
            remaining -= 1

        # Build size_list by repeating each dim_str gpu_counts[dim_str] times
        size_list = []
        for dim_str in sorted(gpu_counts.keys()):  # sorted for determinism
            size_list.extend([dim_str] * gpu_counts[dim_str])

    size_list.sort()
    return size_list


def distribute_dimensions_across_gpus(
    dim_info: dict,
    world_size: int,
    logger: logging.Logger,
) -> list:
    """
    Distribute dimensions across GPUs to balance number of batches per GPU,
    and log the resulting assignment.

    Uses :func:`_compute_size_list` for the allocation logic.

    Args:
        dim_info (dict): Dimension info from analyze_data_distribution.
        world_size (int): Total number of GPUs.
        logger (logging.Logger): Logger.

    Returns:
        list: Sorted size list for each GPU (length = world_size).
    """
    num_dimensions = len(dim_info)
    size_list = _compute_size_list(dim_info, world_size)

    if num_dimensions > world_size:
        logger.info(f"[bucketed_parallel] {num_dimensions} dimensions > {world_size} GPUs")
        logger.info(f"[bucketed_parallel] Selecting top {world_size} dimensions by num_batches")

        logger.info(f"[bucketed_parallel] Selected dimensions (1 GPU each):")
        for i, dim_str in enumerate(size_list):
            info = dim_info[dim_str]
            logger.info(f"  GPU {i}: {dim_str} - {info['count']} files, {info['num_batches']} batches")

        # Warn about skipped dimensions
        selected_set = set(size_list)
        skipped = [d for d in dim_info if d not in selected_set]
        if skipped:
            logger.warning(f"[bucketed_parallel] Skipped dimensions (too few GPUs):")
            for dim_str in skipped:
                info = dim_info[dim_str]
                logger.warning(f"  {dim_str}: {info['count']} files, {info['num_batches']} batches")

        batches_per_gpu = [dim_info[d]['num_batches'] for d in size_list]
        avg = sum(batches_per_gpu) / len(batches_per_gpu)
        max_imbalance = max(abs(b - avg) for b in batches_per_gpu)
        logger.info(f"[bucketed_parallel] Balance: avg={avg:.1f} batches/GPU, max_imbalance={max_imbalance:.1f}")

    else:
        logger.info(f"[bucketed_parallel] {num_dimensions} dimensions <= {world_size} GPUs")
        logger.info(f"[bucketed_parallel] Proportionally allocating GPUs based on batch count")

        # Reconstruct gpu_counts from size_list for logging
        gpu_counts: dict[str, int] = {}
        for d in size_list:
            gpu_counts[d] = gpu_counts.get(d, 0) + 1

        logger.info(f"[bucketed_parallel] GPU allocation per dimension:")
        for dim_str in sorted(gpu_counts.keys(), key=lambda x: dim_info[x]['size_product']):
            n = gpu_counts[dim_str]
            nb = dim_info[dim_str]['num_batches']
            logger.info(
                f"  {dim_str}: {n} GPUs, {nb} total batches, ~{nb / n:.1f} batches/GPU"
            )

        all_loads = [dim_info[d]['num_batches'] / gpu_counts[d] for d in gpu_counts]
        logger.info(
            f"[bucketed_parallel] Balance: max={max(all_loads):.1f}, "
            f"min={min(all_loads):.1f} batches/GPU"
        )

    # Per-GPU assignment table
    gpu_counts_log: dict[str, int] = {}
    for d in size_list:
        gpu_counts_log[d] = gpu_counts_log.get(d, 0) + 1

    logger.info(f"[bucketed_parallel] Per-GPU assignment (sorted by rank):")
    for gpu_id, dim_str in enumerate(size_list):
        n_gpus_for_dim = gpu_counts_log[dim_str]
        nb_total = dim_info[dim_str]['num_batches']
        nb_per_gpu = math.ceil(nb_total / n_gpus_for_dim)
        bs = dim_info[dim_str]['batch_size']
        logger.info(
            f"  GPU {gpu_id:>3}: {dim_str}, batch_size={bs}, ~{nb_per_gpu} batches"
        )

    logger.info(
        f"[bucketed_parallel] Total unique dimensions: {len(dim_info)}, "
        f"GPUs: {world_size}, size_list length: {len(size_list)}"
    )
    return size_list


def _compute_max_iter_from_size_list(dim_info: dict, size_list: list) -> int:
    """
    Compute MAX_ITER as the minimum number of batches-per-GPU across all GPU assignments.

    For each unique dimension in size_list, the per-GPU batch count is:
        ceil(total_batches_for_dim / num_gpus_assigned_to_dim)

    MAX_ITER is the minimum of these values across all dimensions, ensuring every
    GPU completes at least that many iterations before the epoch ends.

    Args:
        dim_info (dict): {dim_str: {'num_batches': int, ...}}
        size_list (list): One entry per GPU (length == world_size).

    Returns:
        int: Dynamic MAX_ITER (>= 1).
    """
    gpu_counts: dict[str, int] = {}
    for d in size_list:
        gpu_counts[d] = gpu_counts.get(d, 0) + 1

    min_batches = min(
        math.ceil(dim_info[d]['num_batches'] / gpu_counts[d])
        for d in gpu_counts
    )
    return max(min_batches, 1)


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
        tuple: (train_files_for_rank, batch_size_for_rank, dimensions_assigned, max_iter)
    """
    # Analyze data distribution (only rank 0 logs details)
    if global_rank == 0:
        dim_info = analyze_data_distribution(all_train_files, base_batch_size, logger)
        size_list = distribute_dimensions_across_gpus(dim_info, world_size, logger)
    else:
        # Other ranks compute the same dim_info, then call _compute_size_list for consistency
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
            batch_size = max(1, min(96, int(base_batch_size * REFERENCE_SIZE_PRODUCT / size_product)))
            num_batches = math.ceil(float(count) / batch_size)
            dim_info[dim_str] = {
                'count': count,
                'batch_size': batch_size,
                'num_batches': num_batches,
                'size_product': size_product,
            }

        # Use the same pure allocation logic as rank 0 (no logging)
        size_list = _compute_size_list(dim_info, world_size)

    size_list.sort()

    # Dynamic MAX_ITER: min batches-per-GPU across all dimension assignments
    max_iter = _compute_max_iter_from_size_list(dim_info, size_list)
    logger.info(f"[bucketed_parallel] Dynamic MAX_ITER = {max_iter} "
                f"(min batches/GPU across all assigned dimensions)")
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
    batch_size = min(batch_size, 96)
    
    logger.info(f"[bucketed_parallel] Rank {global_rank}: calculated batch_size={batch_size} for size {size_str}")
    
    num_batch = float(len(train_files_filtered)) / batch_size
    if math.ceil(num_batch) < max_iter:
        replication_factor = math.ceil(float(max_iter) / num_batch)
        train_files_filtered = train_files_filtered * replication_factor
        logger.info(f"[bucketed_parallel] Rank {global_rank}: replicated data {replication_factor}x "
                   f"to ensure {max_iter} iterations ({len(train_files_filtered)} total files)")

    return train_files_filtered, batch_size, dimensions, max_iter


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, amp: bool = True
) -> None:
    """Main training function with by-size bucketing support."""
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, global_rank, world_size, device = initialize_distributed()
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")

    # Check autoencoder availability and broadcast validation_enabled to all ranks
    setup_validation(args, device, global_rank, logger)

    # TensorBoard: only the global rank-0 process writes events
    tensorboard_writer = None
    if global_rank == 0:
        tensorboard_path = args.tfevent_dir
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        logger.info(f"TensorBoard logging to: {tensorboard_path}")

    # Bucketed-parallel specifics
    logger.info(f"[bucketed_parallel] Using by_size bucketing (script choice determines partition type)")
    logger.info(f"[bucketed_parallel] MAX_ITER will be set dynamically to min(batches/GPU) across all dimensions")
    # image_dim accepts None, a single dim [x,y,z], or a list of dims [[x,y,z],[x2,y2,z2]]
    image_dim = args.diffusion_unet_train.get('image_dim', None)
    if image_dim is not None:
        logger.info(f"[bucketed_parallel] image_dim filter active: {image_dim}")
    else:
        logger.info(f"[bucketed_parallel] image_dim filter: None (all dimensions loaded)")

    # Config summary + checkpoint dir (rank-0 only for filesystem ops)
    logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
    logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
    logger.info(f"[config] data_list -> {args.json_data_list}.")
    logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
    logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
    logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")
    if global_rank == 0:
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

    # Load files — image_dim filter applied inside load_filenames (supports single or multi-dim)
    train_files = load_filenames(args.json_data_list, args.embedding_base_dir, image_dim=image_dim)
    logger.info(f"num_files_train: {len(train_files)}")

    # Filter missing files, then compute modality-based sampling weights
    valid_train_files = filter_existing_files(train_files, logger)
    compute_modality_weights(valid_train_files, max_weight=10.0, logger=logger)

    # By-size bucketing: partition data by image dimensions across GPUs
    partitioned_train_files, batch_size, assigned_dims, max_iter = partition_data_by_size(
        all_train_files=valid_train_files,
        global_rank=global_rank,
        world_size=world_size,
        logger=logger,
        base_batch_size=args.diffusion_unet_train["batch_size"],
    )
    logger.info(f"[bucketed_parallel] Rank {global_rank}: {len(partitioned_train_files)} files, "
                f"batch_size={batch_size}, dims={assigned_dims}, max_iter={max_iter}")

    use_weighted_sampling = args.diffusion_unet_train.get("use_weighted_sampling", True)
    logger.info(f"Weighted sampling: {'enabled' if use_weighted_sampling else 'disabled'}")

    train_loader, sampler = prepare_data(
        partitioned_train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=batch_size,
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping=args.modality_mapping,
        use_weighted_sampling=use_weighted_sampling,
    )

    if scale_factor is None:
        scale_factor = calculate_scale_factor(train_loader, device, logger)
    else:
        logger.info(f"Using loaded scale_factor: {scale_factor}.")

    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    if start_epoch > 0 and optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info("Loaded optimizer state from checkpoint.")
    elif optimizer_state_dict is not None:
        logger.warning("Skipping optimizer state loading.")

    # Total steps for by_size bucketing: one step per epoch (scheduler steps per epoch)
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
            batch_size,
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            global_rank,
            amp=amp,
            max_iter=max_iter,  # Dynamic: min batches/GPU across all assigned dimensions
        )
        lr_scheduler.step()  # Step LR once per epoch

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or global_rank == 0:
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

            if args.validation_enabled and (epoch + 1) % VAL_INTERVAL == 0:
                logger.info(f"Generating validation images at epoch {epoch + 1}...")
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
