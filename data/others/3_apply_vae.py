#!/usr/bin/env python3
"""
Script to apply trained VAE (autoencoder) to skull-stripped MR images.
Generic version for datasets like AOMIC and QTIM.

This script:
1. Loads input JSON with training/testing splits
2. Applies preprocessing (orientation, intensity normalization)
3. Resamples to target sizes
4. Encodes images using VAE to get latent embeddings
5. Saves embeddings as NIfTI files

Uses shared utilities from vae_utils.py
Does NOT generate output JSON - use 4_create_json_mr_emb.py for that.
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


def main():
    parser = argparse.ArgumentParser(description="Apply VAE to skull-stripped MR images (Generic)")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to input JSON file (with training/testing splits)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory where image paths in JSON are relative to")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for output embeddings")
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
        print("Generic VAE Application (using vae_utils)")
        print("="*80)
        print(f"Input JSON: {args.input_json}")
        print(f"Data root: {args.data_root}")
        print(f"Output root: {args.output_root}")
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
    
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # Combine training, testing, and validation images
    all_images = []
    if "training" in data:
        all_images.extend(data["training"])
    if "testing" in data:
        all_images.extend(data["testing"])
    if "validation" in data:
        all_images.extend(data["validation"])
    
    if global_rank == 0:
        print(f"Total images in JSON: {len(all_images)}")
        if "training" in data:
            print(f"  Training: {len(data['training'])}")
        if "testing" in data:
            print(f"  Testing: {len(data['testing'])}")
        if "validation" in data:
            print(f"  Validation: {len(data['validation'])}")
    
    # Distribute images across GPUs
    images_per_gpu = len(all_images) // world_size
    start_idx = local_rank * images_per_gpu
    end_idx = start_idx + images_per_gpu if local_rank < world_size - 1 else len(all_images)
    my_images = all_images[start_idx:end_idx]
    
    if global_rank == 0:
        print(f"GPU {local_rank}: Processing images {start_idx} to {end_idx-1} ({len(my_images)} images)")
        print("="*80)
    
    # Process images
    total_success = 0
    total_errors = 0
    total_skipped = 0
    all_error_messages = []
    
    if global_rank == 0:
        print("\n" + "="*80)
        print("Processing images...")
        print("="*80)
    
    for item in tqdm(my_images, desc=f"GPU {global_rank}", disable=(global_rank != 0)):
        # Extract image path from item
        if isinstance(item, dict):
            image_rel_path = item.get("image", "")
        else:
            image_rel_path = str(item)
        
        if not image_rel_path:
            all_error_messages.append("Empty image path in JSON item")
            total_errors += 1
            continue
        
        # Build full path
        image_full_path = os.path.join(args.data_root, image_rel_path)
        
        # Use shared process_single_image function
        num_success, num_errors, num_skipped, error_messages = process_single_image(
            image_full_path=image_full_path,
            image_rel_path=image_rel_path,
            output_root=args.output_root,
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
        print(f"GPU {local_rank} Summary:")
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
