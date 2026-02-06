#!/usr/bin/env python3
"""
Script to apply brain masks to MR images (skull stripping) - TEST VERSION.

This script reads image-brain_mask pairs from the JSON file, multiplies
the image by the brain mask to perform skull stripping, and saves the
result to a new directory with the same structure.

This is a TEST version that processes only ONE SUBJECT for verification.

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

# TEST MODE: Process only one subject
TEST_SUBJECT_ID = "3159398"  # First subject


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
        print(f"  Loading image: {os.path.basename(image_path)}")
        img_nii = nib.load(image_path)
        img_data = img_nii.get_fdata()
        print(f"    Image shape: {img_data.shape}, dtype: {img_data.dtype}")
        print(f"    Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
        
        # Load brain mask
        print(f"  Loading mask: {os.path.basename(mask_path)}")
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        print(f"    Mask shape: {mask_data.shape}, dtype: {mask_data.dtype}")
        print(f"    Mask range: [{mask_data.min():.2f}, {mask_data.max():.2f}]")
        print(f"    Mask unique values: {np.unique(mask_data)}")
        
        # Check if shapes match
        if img_data.shape != mask_data.shape:
            print(f"ERROR: Shape mismatch!")
            print(f"  Image shape: {img_data.shape}, Mask shape: {mask_data.shape}")
            return False
        
        # Apply brain mask (element-wise multiplication)
        print(f"  Applying brain mask (multiplication)...")
        skull_stripped = img_data * mask_data
        print(f"    Output range: [{skull_stripped.min():.2f}, {skull_stripped.max():.2f}]")
        
        # Calculate percentage of non-zero voxels
        non_zero_original = np.count_nonzero(img_data)
        non_zero_masked = np.count_nonzero(skull_stripped)
        print(f"    Non-zero voxels: {non_zero_masked}/{non_zero_original} ({100*non_zero_masked/non_zero_original:.1f}%)")
        
        # Create output NIfTI image with same header/affine as original
        output_nii = nib.Nifti1Image(skull_stripped, img_nii.affine, img_nii.header)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save skull-stripped image
        print(f"  Saving to: {output_path}")
        nib.save(output_nii, output_path)
        print(f"  ✓ Success!")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_dataset():
    """
    Main function to process all image-brain_mask pairs for ONE subject.
    """
    # Load JSON data
    pairs = load_json(INPUT_JSON)
    
    # Filter pairs for test subject only
    test_pairs = [p for p in pairs if p['subject_id'] == TEST_SUBJECT_ID]
    
    print("=" * 80)
    print(f"TEST MODE: Processing only subject {TEST_SUBJECT_ID}")
    print(f"Found {len(test_pairs)} image-brain_mask pairs for this subject")
    print(f"Input data root: {DATA_ROOT}")
    print(f"Output data root: {OUTPUT_ROOT}")
    print("=" * 80)
    
    if len(test_pairs) == 0:
        print(f"ERROR: No pairs found for subject {TEST_SUBJECT_ID}")
        return
    
    # Statistics
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each pair
    print("\nProcessing image-brain_mask pairs...")
    print("=" * 80)
    for idx, pair in enumerate(test_pairs):
        print(f"\n[{idx+1}/{len(test_pairs)}] Processing pair:")
        print(f"  Subject: {pair['subject_id']}")
        print(f"  Space: {pair['space']}")
        print(f"  Modality: {pair.get('mr_contrast_weighting', 'N/A')} - {pair.get('acquisition_plane', 'N/A')}")
        
        # Get relative paths from JSON
        image_rel_path = pair['image']
        mask_rel_path = pair['brain_mask']
        
        # Construct full paths
        image_full_path = os.path.join(DATA_ROOT, image_rel_path)
        mask_full_path = os.path.join(DATA_ROOT, mask_rel_path)
        output_full_path = os.path.join(OUTPUT_ROOT, image_rel_path)
        
        # Check if input files exist
        if not os.path.exists(image_full_path):
            print(f"ERROR: Image not found: {image_full_path}")
            failed_count += 1
            continue
            
        if not os.path.exists(mask_full_path):
            print(f"ERROR: Mask not found: {mask_full_path}")
            failed_count += 1
            continue
        
        # Skip if output already exists
        if os.path.exists(output_full_path):
            print(f"  Skipped (output already exists)")
            skipped_count += 1
            continue
        
        # Apply brain mask
        success = apply_brain_mask(image_full_path, mask_full_path, output_full_path)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        print("-" * 80)
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Processing Summary ===")
    print("=" * 80)
    print(f"Subject: {TEST_SUBJECT_ID}")
    print(f"Total pairs for this subject: {len(test_pairs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)
    
    # Show sample output paths
    if success_count > 0:
        print(f"\nSkull-stripped images saved to:")
        print(f"  {OUTPUT_ROOT}/{TEST_SUBJECT_ID}/")
        print("\nYou can now visually inspect these images to verify the skull stripping.")
        print("\nIf results look good, run the full version:")
        print("  python ./data/2_apply_brain_mask.py")
    
    print("\nTest processing complete!")


if __name__ == "__main__":
    process_dataset()
