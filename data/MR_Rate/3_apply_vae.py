#!/usr/bin/env python3
"""
Script to apply trained VAE (autoencoder) to skull-stripped MR images.
MR-RATE dataset specific version.

This script:
1. Downloads the autoencoder model (models/autoencoder_v1.pt)
2. Loads skull-stripped MRI images
3. Applies preprocessing (orientation, intensity normalization)
4. Resamples to target sizes
5. Encodes images using VAE to get latent embeddings
6. Saves embeddings as NIfTI files

Uses shared utilities from vae_utils.py
Based on logic from: diffusion_model_scale_up/data/1_resample_128_v1_1.py
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from monai.utils import set_determinism

# Add parent directory to path for vae_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from vae_utils import (
    download_autoencoder_model,
    load_autoencoder,
    create_preprocessing_transforms,
    process_single_image,
    POSSIBLE_SIZES
)

# Configuration
INPUT_JSON = "./jsons/dataset_MR-RATE_brain_mask_pairs.json"
SKULL_STRIPPED_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/hyperlinks_removed/datasets--Forithmus--MR-RATE/snapshots/6c419668310c03d150b7904821a5b41ed1123318/nvidia_1000_mri_skull_stripped"
OUTPUT_EMBEDDING_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/MR-Rate/nvidia_1000_mri_skull_stripped"


def main():
    parser = argparse.ArgumentParser(description="Apply VAE to skull-stripped MR images (MR-RATE)")
    parser.add_argument("--num-splits", "--num_splits", dest="num_splits", type=int, default=1,
                        help="Number of splits for tensor parallelism (1, 2, 4, 8, or 16)")
    parser.add_argument("--no_skip_existing", "--no-skip-existing", dest="no_skip_existing", action="store_true",
                        help="Reprocess all files (don't skip existing)")
    args = parser.parse_args()
    
    # Initialize distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
    
    dist.init_process_group(backend="nccl", init_method="env://")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    # For multi-node: local_rank is GPU ID on current node (0-7)
    # global_rank is overall rank across all nodes (0-31 for 4 nodes × 8 GPUs)
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    set_determinism(seed=42)
    
    skip_existing = not args.no_skip_existing
    
    if global_rank == 0:
        print("="*80)
        print("MR-RATE VAE Application (using vae_utils)")
        print("="*80)
        print(f"Input JSON: {INPUT_JSON}")
        print(f"Skull-stripped root: {SKULL_STRIPPED_ROOT}")
        print(f"Output embedding root: {OUTPUT_EMBEDDING_ROOT}")
        print(f"Num splits: {args.num_splits}")
        print(f"Skip existing: {skip_existing}")
        print(f"World size: {world_size}")
        print("="*80)
    
    # Download model (only on rank 0)
    download_autoencoder_model(local_rank=global_rank)
    dist.barrier()  # Wait for download to complete
    
    # Load autoencoder
    if global_rank == 0:
        print("\n" + "="*80)
        print("Loading autoencoder...")
        print("="*80)
    autoencoder = load_autoencoder(device=device, num_splits=args.num_splits)
    
    # Create preprocessing transforms
    transforms = create_preprocessing_transforms()
    
    # Load input JSON
    if global_rank == 0:
        print("\n" + "="*80)
        print("Loading input JSON...")
        print("="*80)
    
    with open(INPUT_JSON, 'r') as f:
        pairs = json.load(f)
    
    if global_rank == 0:
        print(f"Total pairs in JSON: {len(pairs)}")
    
    # Distribute pairs across GPUs
    pairs_per_gpu = len(pairs) // world_size
    start_idx = global_rank * pairs_per_gpu
    end_idx = start_idx + pairs_per_gpu if global_rank < world_size - 1 else len(pairs)
    my_pairs = pairs[start_idx:end_idx]
    
    if global_rank == 0:
        print(f"GPU {global_rank}: Processing pairs {start_idx} to {end_idx-1} ({len(my_pairs)} pairs)")
        print("="*80)
    
    # Process pairs
    total_success = 0
    total_errors = 0
    total_skipped = 0
    all_error_messages = []
    
    if global_rank == 0:
        print("\n" + "="*80)
        print("Processing images...")
        print("="*80)
    
    for pair in tqdm(my_pairs, desc=f"GPU {global_rank}", disable=(global_rank != 0)):
        # Extract image relative path from pair
        image_rel_path = pair["image"]
        image_full_path = os.path.join(SKULL_STRIPPED_ROOT, image_rel_path)
        
        # Use shared process_single_image function
        num_success, num_errors, num_skipped, error_messages = process_single_image(
            image_full_path=image_full_path,
            image_rel_path=image_rel_path,
            output_root=OUTPUT_EMBEDDING_ROOT,
            transforms=transforms,
            autoencoder=autoencoder,
            device=device,
            num_splits=args.num_splits,
            skip_existing=skip_existing
        )
        
        total_success += num_success
        total_errors += num_errors
        total_skipped += num_skipped
        all_error_messages.extend(error_messages)
    
    # Print summary for this GPU
    if global_rank == 0:
        print("\n" + "="*80)
        print(f"GPU {global_rank} Summary:")
        print("="*80)
        print(f"Successfully encoded: {total_success}")
        print(f"Skipped (existing/too large): {total_skipped}")
        print(f"Errors: {total_errors}")
        
        if all_error_messages:
            print("\nError messages:")
            for msg in all_error_messages[:10]:  # Show first 10 errors
                print(f"  - {msg}")
            if len(all_error_messages) > 10:
                print(f"  ... and {len(all_error_messages) - 10} more errors")
        
        print("="*80)
        print("Done!")
        print("="*80)
    
    # Don't call destroy_process_group() - it can hang in multi-node setups
    # The process group will be cleaned up when the script exits


if __name__ == "__main__":
    main()
