#!/usr/bin/env python3
"""
Script to apply brain masks to MR images (skull stripping).

This script reads image-brain_mask pairs from the JSON file, multiplies
the image by the brain mask to perform skull stripping, and saves the
result to a new directory with the same structure.

Input: ./jsons/dataset_MR-RATE_brain_mask_pairs.json
Output: nvidia_1000_mri_skull_stripped/ directory with same structure
"""

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
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


def apply_brain_mask(image_path, mask_path, output_path):
    """
    Apply brain mask to image by multiplication (skull stripping).
    
    Args:
        image_path (str): Path to input image
        mask_path (str): Path to brain mask
        output_path (str): Path to save skull-stripped image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load image
        img_nii = nib.load(image_path)
        img_data = img_nii.get_fdata()
        
        # Load brain mask
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        
        # Check if shapes match
        if img_data.shape != mask_data.shape:
            print(f"Warning: Shape mismatch for {image_path}")
            print(f"  Image shape: {img_data.shape}, Mask shape: {mask_data.shape}")
            return False
        
        # Apply brain mask (element-wise multiplication)
        skull_stripped = img_data * mask_data
        
        # Create output NIfTI image with same header/affine as original
        output_nii = nib.Nifti1Image(skull_stripped, img_nii.affine, img_nii.header)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save skull-stripped image
        nib.save(output_nii, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_dataset():
    """
    Main function to process all image-brain_mask pairs.
    """
    # Load JSON data
    pairs = load_json(INPUT_JSON)
    
    print("=" * 80)
    print(f"Input data root: {DATA_ROOT}")
    print(f"Output data root: {OUTPUT_ROOT}")
    print("=" * 80)
    
    # Statistics
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each pair with progress bar
    print("\nProcessing image-brain_mask pairs...")
    for pair in tqdm(pairs, desc="Skull stripping", unit="image"):
        # Get relative paths from JSON
        image_rel_path = pair['image']
        mask_rel_path = pair['brain_mask']
        
        # Construct full paths
        image_full_path = os.path.join(DATA_ROOT, image_rel_path)
        mask_full_path = os.path.join(DATA_ROOT, mask_rel_path)
        output_full_path = os.path.join(OUTPUT_ROOT, image_rel_path)
        
        # Check if input files exist
        if not os.path.exists(image_full_path):
            print(f"Warning: Image not found: {image_full_path}")
            failed_count += 1
            continue
            
        if not os.path.exists(mask_full_path):
            print(f"Warning: Mask not found: {mask_full_path}")
            failed_count += 1
            continue
        
        # Skip if output already exists (optional - comment out to overwrite)
        if os.path.exists(output_full_path):
            skipped_count += 1
            continue
        
        # Apply brain mask
        success = apply_brain_mask(image_full_path, mask_full_path, output_full_path)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Processing Summary ===")
    print("=" * 80)
    print(f"Total pairs processed: {len(pairs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)
    
    # Show output directory structure
    if success_count > 0:
        print(f"\nSkull-stripped images saved to:")
        print(f"  {OUTPUT_ROOT}")
        print("\nDirectory structure maintained from original dataset.")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    process_dataset()
