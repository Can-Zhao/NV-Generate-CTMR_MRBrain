#!/usr/bin/env python3
"""
Script to apply brain masks to MR images (skull stripping) - PARALLEL VERSION.

This script uses multiprocessing to accelerate the skull stripping process
by processing multiple images in parallel across multiple CPU cores.

Input: ./jsons/dataset_MR-RATE_brain_mask_pairs.json
Output: nvidia_1000_mri_skull_stripped/ directory with same structure

Usage:
    python ./data/MR_Rate/2_apply_brain_mask_parallel.py --num-workers 8
"""

import os
import json
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
from tqdm import tqdm

# Configuration
INPUT_JSON = "./jsons/dataset_MR-RATE_brain_mask_pairs.json"
DATA_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/hyperlinks_removed/datasets--Forithmus--MR-RATE/snapshots/6c419668310c03d150b7904821a5b41ed1123318/nvidia_1000_mri"
OUTPUT_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/hyperlinks_removed/datasets--Forithmus--MR-RATE/snapshots/6c419668310c03d150b7904821a5b41ed1123318/nvidia_1000_mri_skull_stripped"


def load_json(json_path):
    """
    Load the dataset JSON file.
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        list: List of image-brain_mask pairs
    """
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} image-brain_mask pairs")
    return data


def apply_brain_mask_single(pair, data_root, output_root, skip_existing=True):
    """
    Apply brain mask to a single image (worker function for multiprocessing).
    
    Args:
        pair (dict): Dictionary containing image and brain_mask paths
        data_root (str): Root directory of input data
        output_root (str): Root directory of output data
        skip_existing (bool): Whether to skip already processed images
        
    Returns:
        tuple: (success: bool, message: str, pair_info: dict)
    """
    try:
        # Get relative paths from JSON
        image_rel_path = pair['image']
        mask_rel_path = pair['brain_mask']
        
        # Construct full paths
        image_full_path = os.path.join(data_root, image_rel_path)
        mask_full_path = os.path.join(data_root, mask_rel_path)
        output_full_path = os.path.join(output_root, image_rel_path)
        
        # Check if input files exist
        if not os.path.exists(image_full_path):
            return (False, f"Image not found: {image_full_path}", pair)
            
        if not os.path.exists(mask_full_path):
            return (False, f"Mask not found: {mask_full_path}", pair)
        
        # Skip if output already exists AND is valid
        if skip_existing and os.path.exists(output_full_path):
            # Verify the existing file is valid (not corrupted)
            # Just check if file can be loaded, don't load full data (expensive!)
            try:
                test_nii = nib.load(output_full_path)
                # If we can load the header successfully, file is likely valid - skip it
                return (None, "Already exists (verified)", pair)  # None = skipped
            except Exception as e:
                # File exists but is corrupted - reprocess it
                pass  # Continue to reprocess
        
        # Load image
        img_nii = nib.load(image_full_path)
        img_data = img_nii.get_fdata()
        
        # Load brain mask
        mask_nii = nib.load(mask_full_path)
        mask_data = mask_nii.get_fdata()
        
        # Check if shapes match
        if img_data.shape != mask_data.shape:
            return (False, f"Shape mismatch: img {img_data.shape} vs mask {mask_data.shape}", pair)
        
        # Apply brain mask (element-wise multiplication)
        skull_stripped = img_data * mask_data
        
        # Create output NIfTI image with same header/affine as original
        output_nii = nib.Nifti1Image(skull_stripped, img_nii.affine, img_nii.header)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        
        # Save skull-stripped image using atomic write (write to temp file, then rename)
        # This prevents corrupted files if process is interrupted during save
        # Use .tmp.nii.gz extension so nibabel can recognize it as a valid NIfTI file
        temp_output_path = output_full_path.replace('.nii.gz', '.tmp.nii.gz')
        try:
            nib.save(output_nii, temp_output_path)
            # Atomic rename - if this succeeds, file is complete
            os.replace(temp_output_path, output_full_path)
        except Exception as e:
            # Clean up temp file if save failed
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise e
        
        return (True, "Success", pair)
        
    except Exception as e:
        return (False, f"Error: {str(e)}", pair)


def process_dataset_parallel(num_workers=None, skip_existing=True):
    """
    Main function to process all image-brain_mask pairs using multiprocessing.
    
    Args:
        num_workers (int): Number of parallel workers. If None, uses cpu_count()
        skip_existing (bool): Whether to skip already processed images
    """
    # Load JSON data
    pairs = load_json(INPUT_JSON)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print("=" * 80)
    print(f"PARALLEL PROCESSING MODE")
    print(f"Number of workers: {num_workers}")
    print(f"Total pairs to process: {len(pairs)}")
    print(f"Skip existing: {skip_existing}")
    print(f"Input data root: {DATA_ROOT}")
    print(f"Output data root: {OUTPUT_ROOT}")
    print("=" * 80)
    
    # Create partial function with fixed arguments
    worker_func = partial(
        apply_brain_mask_single,
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        skip_existing=skip_existing
    )
    
    # Statistics
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_pairs = []
    
    # Process with multiprocessing pool
    print("\nProcessing image-brain_mask pairs in parallel...")
    with Pool(processes=num_workers) as pool:
        # Use imap for better progress tracking
        results = list(tqdm(
            pool.imap(worker_func, pairs),
            total=len(pairs),
            desc="Skull stripping",
            unit="image"
        ))
    
    # Collect statistics
    for success, message, pair in results:
        if success is True:
            success_count += 1
        elif success is False:
            failed_count += 1
            failed_pairs.append((pair, message))
        else:  # success is None (skipped)
            skipped_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Processing Summary ===")
    print("=" * 80)
    print(f"Total pairs: {len(pairs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Workers used: {num_workers}")
    print("=" * 80)
    
    # Show failed pairs if any
    if failed_pairs:
        print("\n⚠️  Failed pairs:")
        for pair, error_msg in failed_pairs[:10]:  # Show first 10
            print(f"  Subject: {pair['subject_id']}, Space: {pair['space']}, "
                  f"Modality: {pair.get('modality_key', 'N/A')}")
            print(f"    Error: {error_msg}")
        if len(failed_pairs) > 10:
            print(f"  ... and {len(failed_pairs) - 10} more failed pairs")
    
    # Show output directory structure
    if success_count > 0:
        print(f"\n✓ Skull-stripped images saved to:")
        print(f"  {OUTPUT_ROOT}")
        print("\nDirectory structure maintained from original dataset.")
    
    print("\nProcessing complete!")
    
    # Return statistics
    return {
        'total': len(pairs),
        'success': success_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'failed_pairs': failed_pairs
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply brain masks to MR images using parallel processing'
    )
    parser.add_argument(
        '--num-workers', '-n',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Reprocess all images even if output exists'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Available CPU cores: {cpu_count()}")
    if args.num_workers is None:
        print(f"Using all available cores: {cpu_count()}")
    
    stats = process_dataset_parallel(
        num_workers=args.num_workers,
        skip_existing=not args.no_skip_existing
    )
