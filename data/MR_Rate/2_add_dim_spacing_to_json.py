#!/usr/bin/env python3
"""
Script to add 'dim' and 'spacing' information to the MR-RATE dataset JSON file.

For each image:
1. Loads the NIfTI file
2. Extracts 3D dimensions and voxel spacing
3. Adds 'dim' and 'spacing' keys to each entry in the JSON file

This script should be run first before running the analysis script.
"""

import os
import json
import nibabel as nib
from multiprocessing import Pool
from functools import partial

# Configuration
DATA_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri/"
JSON_FILE = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_brain_mask_pairs.json"
NUM_WORKERS = 96


def extract_dim_and_spacing(image_path):
    """
    Extract 3D dimensions and spacing from a NIfTI image.
    
    Args:
        image_path (str): Full path to the NIfTI image file
        
    Returns:
        tuple: (dims, spacing) as lists, or (None, None) if error
    """
    if not os.path.exists(image_path):
        return None, None
    
    try:
        # Load the image and reorient to RAS (same as MONAI does in 4_apply_vae.py)
        # so that dim and spacing axes match the RAS convention [X, Y, Z].
        # Without this, get_zooms() returns spacing in native storage order,
        # which may differ from RAS order for non-canonical orientations.
        img = nib.as_closest_canonical(nib.load(image_path))

        # Get dimensions in RAS order (cast to Python int for JSON serialization)
        dims = [int(d) for d in img.shape[:3]]

        # Get voxel spacing in RAS order after reorientation (cast to Python float
        # because nibabel returns numpy.float32 which json.dump cannot serialize)
        spacing = [float(s) for s in img.header.get_zooms()[:3]]

        return dims, spacing
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def add_dim_spacing_to_json():
    """
    Main function to add dim and spacing to the JSON file.
    """
    # Load the JSON file
    print("Loading JSON file...")
    with open(JSON_FILE, 'r') as f:
        image_pairs = json.load(f)
    
    print(f"Found {len(image_pairs)} image-mask pairs")
    print("=" * 80)
    
    # Separate entries that need processing from those already done
    to_process = []
    skipped_indices = []
    for idx, pair in enumerate(image_pairs):
        if 'dim' in pair and 'spacing' in pair:
            skipped_indices.append(idx)
        else:
            to_process.append((idx, os.path.join(DATA_ROOT, pair.get('image', ''))))

    skipped_count = len(skipped_indices)
    print(f"Skipping {skipped_count} entries (already have dim/spacing)")
    print(f"Processing {len(to_process)} entries with {NUM_WORKERS} workers...")

    # Extract dim/spacing in parallel
    image_paths = [path for _, path in to_process]
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(extract_dim_and_spacing, image_paths)

    # Write results back into image_pairs
    updated_count = 0
    for (idx, _), (dims, spacing) in zip(to_process, results):
        if dims is not None and spacing is not None:
            image_pairs[idx]['dim'] = dims
            image_pairs[idx]['spacing'] = spacing
            updated_count += 1
        else:
            print(f"Warning: Failed to extract dim/spacing for {image_pairs[idx].get('image', '')}")

    print(f"\nProcessed {len(to_process)} images")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (already had dim/spacing): {skipped_count}")
    print("=" * 80)
    
    # Save updated JSON file
    print(f"\nSaving updated JSON file to {JSON_FILE}...")
    with open(JSON_FILE, 'w') as f:
        json.dump(image_pairs, f, indent=2)
    print(f"Updated JSON file saved successfully!")
    print("=" * 80)


if __name__ == "__main__":
    add_dim_spacing_to_json()
