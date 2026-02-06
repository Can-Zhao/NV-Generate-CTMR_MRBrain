#!/usr/bin/env python3
"""
Script to generate image-brain_mask pair JSON for MR-RATE dataset.

This script processes the MR-RATE dataset and creates a JSON file containing
image/brain_mask pairs with associated metadata for each subject.

Processing logic:
1. Native space: 1-to-1 matching between images and brain_masks
2. Atlas space: 1-to-many matching (single brain_mask paired with all images)

Output: ./jsons/dataset_MR-RATE.json
"""

import os
import json
import glob
from pathlib import Path

# Configuration
DATA_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/hyperlinks_removed/datasets--Forithmus--MR-RATE/snapshots/6c419668310c03d150b7904821a5b41ed1123318/nvidia_1000_mri"
OUTPUT_JSON = "./jsons/dataset_MR-RATE_brain_mask_pairs.json"


def extract_modality_key(filename):
    """
    Extract modality key from filename.
    
    Examples:
        'native__flair-raw-axi-nc.nii.gz' -> 'flair-raw-axi-nc'
        'atlas__t1-raw-cor-nc.nii.gz' -> 't1-raw-cor-nc'
    
    Args:
        filename (str): Filename to extract modality key from
        
    Returns:
        str: Extracted modality key
    """
    basename = os.path.basename(filename)
    basename = basename.replace('.nii.gz', '')
    
    if basename.startswith('native__'):
        return basename[8:]  # Remove 'native__'
    elif basename.startswith('atlas__'):
        return basename[7:]   # Remove 'atlas__'
    else:
        return basename


def get_relative_path(full_path, data_root):
    """
    Convert absolute path to relative path starting from subject_id.
    
    Example:
        Input: /data_root/3159398/native_space/img/native__flair-raw-axi-nc.nii.gz
        Output: 3159398/native_space/img/native__flair-raw-axi-nc.nii.gz
    
    Args:
        full_path (str): Full absolute path
        data_root (str): Root directory of the dataset
        
    Returns:
        str: Relative path starting from subject_id
    """
    rel_path = os.path.relpath(full_path, data_root)
    return rel_path


def load_metadata(subject_dir):
    """
    Load and parse metadata.json from subject directory.
    
    Args:
        subject_dir (str): Path to subject directory
        
    Returns:
        dict: Combined metadata from center_modality and moving_modality
    """
    metadata_path = os.path.join(subject_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
        return {}
    
    # Combine center_modality and moving_modality into one dict
    all_modalities = {}
    if "center_modality" in metadata:
        all_modalities.update(metadata["center_modality"])
    if "moving_modality" in metadata:
        all_modalities.update(metadata["moving_modality"])
    
    return all_modalities


def process_native_space(subject_dir, subject_id, metadata_dict, data_root):
    """
    Process native_space: 1-to-1 matching between images and brain_masks.
    
    For each image in native_space/img/, find the corresponding brain_mask
    in native_space/seg/ by adding '_brain_mask' suffix.
    
    Args:
        subject_dir (str): Path to subject directory
        subject_id (str): Subject ID
        metadata_dict (dict): Metadata dictionary for this subject
        data_root (str): Root directory of the dataset
        
    Returns:
        list: List of data pair dictionaries
    """
    pairs = []
    
    img_dir = os.path.join(subject_dir, "native_space", "img")
    seg_dir = os.path.join(subject_dir, "native_space", "seg")
    
    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(seg_dir):
        return pairs
    
    # Get all .nii.gz files in img directory
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    
    for image_path in image_files:
        # Construct corresponding brain_mask path
        image_basename = os.path.basename(image_path)
        mask_basename = image_basename.replace('.nii.gz', '_brain_mask.nii.gz')
        mask_path = os.path.join(seg_dir, mask_basename)
        
        # Check if brain_mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Brain mask not found for {image_path}")
            continue
        
        # Extract modality key
        modality_key = extract_modality_key(image_basename)
        
        # Get metadata for this modality
        modality_metadata = metadata_dict.get(modality_key, {})
        
        # Convert to relative paths
        image_rel_path = get_relative_path(image_path, data_root)
        mask_rel_path = get_relative_path(mask_path, data_root)
        
        # Create data pair entry
        pair = {
            "image": image_rel_path,
            "brain_mask": mask_rel_path,
            "subject_id": subject_id,
            "space": "native",
            "modality_key": modality_key,
        }
        
        # Add all metadata fields
        pair.update(modality_metadata)
        
        pairs.append(pair)
    
    return pairs


def deduplicate_atlas_images(image_files):
    """
    Deduplicate atlas space images that differ only in acquisition plane.
    
    In atlas space, images registered to 1x1x1mm spacing with different
    acquisition planes (cor/sag/axi) are essentially the same. We keep only
    one version (preferring axi > cor > sag).
    
    Args:
        image_files (list): List of image file paths
        
    Returns:
        list: Deduplicated list of image file paths
    """
    from collections import defaultdict
    
    # Group images by their base name (without acquisition plane)
    groups = defaultdict(list)
    
    for image_path in image_files:
        basename = os.path.basename(image_path)
        
        # Create a normalized key by removing acquisition plane indicators
        # Replace -cor-, -sag-, -axi- with a placeholder
        normalized = basename.replace('-cor-', '-PLANE-').replace('-sag-', '-PLANE-').replace('-axi-', '-PLANE-')
        
        groups[normalized].append(image_path)
    
    # For each group, keep only one image (prefer axi > cor > sag)
    deduplicated = []
    
    for normalized_key, images in groups.items():
        if len(images) == 1:
            # No duplicates, keep as is
            deduplicated.append(images[0])
        else:
            # Multiple images with different planes, pick one
            # Priority: axi > cor > sag
            axi_images = [img for img in images if '-axi-' in os.path.basename(img)]
            cor_images = [img for img in images if '-cor-' in os.path.basename(img)]
            sag_images = [img for img in images if '-sag-' in os.path.basename(img)]
            
            if axi_images:
                deduplicated.append(axi_images[0])
            elif cor_images:
                deduplicated.append(cor_images[0])
            elif sag_images:
                deduplicated.append(sag_images[0])
            else:
                # Fallback: just take the first one
                deduplicated.append(images[0])
    
    return sorted(deduplicated)


def process_atlas_space(subject_dir, subject_id, metadata_dict, data_root):
    """
    Process atlas_space: 1-to-many matching.
    
    Find the single brain_mask in atlas_space/seg/ and pair it with
    all images in atlas_space/img/.
    
    Note: Atlas space images with different acquisition planes (cor/sag/axi)
    are deduplicated as they represent the same registered image.
    
    Args:
        subject_dir (str): Path to subject directory
        subject_id (str): Subject ID
        metadata_dict (dict): Metadata dictionary for this subject
        data_root (str): Root directory of the dataset
        
    Returns:
        list: List of data pair dictionaries
    """
    pairs = []
    
    img_dir = os.path.join(subject_dir, "atlas_space", "img")
    seg_dir = os.path.join(subject_dir, "atlas_space", "seg")
    
    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(seg_dir):
        return pairs
    
    # Find the brain_mask file (should be only one)
    brain_mask_files = glob.glob(os.path.join(seg_dir, "*brain_mask.nii.gz"))
    
    if len(brain_mask_files) == 0:
        print(f"Warning: No brain_mask found in {seg_dir}")
        return pairs
    
    if len(brain_mask_files) > 1:
        print(f"Warning: Multiple brain_masks found in {seg_dir}, using first one")
    
    brain_mask_path = brain_mask_files[0]
    
    # Get all .nii.gz files in img directory
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    
    # Deduplicate images that differ only in acquisition plane
    image_files = deduplicate_atlas_images(image_files)
    
    for image_path in image_files:
        image_basename = os.path.basename(image_path)
        
        # Extract modality key
        modality_key = extract_modality_key(image_basename)
        
        # Get metadata for this modality
        modality_metadata = metadata_dict.get(modality_key, {})
        
        # Convert to relative paths
        image_rel_path = get_relative_path(image_path, data_root)
        mask_rel_path = get_relative_path(brain_mask_path, data_root)
        
        # Create data pair entry
        pair = {
            "image": image_rel_path,
            "brain_mask": mask_rel_path,  # Same mask for all images
            "subject_id": subject_id,
            "space": "atlas",
            "modality_key": modality_key,
        }
        
        # Add all metadata fields
        pair.update(modality_metadata)
        
        pairs.append(pair)
    
    return pairs


def generate_dataset_json():
    """
    Main function to generate dataset JSON file.
    
    Processes all subjects in the MR-RATE dataset and creates a JSON file
    containing image/brain_mask pairs with metadata.
    """
    all_pairs = []
    
    # Get all subject directories
    subject_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
    
    print(f"Found {len(subject_dirs)} subjects in {DATA_ROOT}")
    print("=" * 80)
    
    # Process each subject
    for idx, subject_dir in enumerate(subject_dirs):
        subject_id = os.path.basename(subject_dir)
        
        if (idx + 1) % 50 == 0:
            print(f"Processing subject {idx + 1}/{len(subject_dirs)}: {subject_id}")
        
        try:
            # Load metadata for this subject
            metadata_dict = load_metadata(subject_dir)
            
            # Process native_space
            native_pairs = process_native_space(subject_dir, subject_id, metadata_dict, DATA_ROOT)
            all_pairs.extend(native_pairs)
            
            # Process atlas_space
            atlas_pairs = process_atlas_space(subject_dir, subject_id, metadata_dict, DATA_ROOT)
            all_pairs.extend(atlas_pairs)
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save to JSON file
    print("=" * 80)
    print(f"Total pairs found: {len(all_pairs)}")
    print(f"Saving to {OUTPUT_JSON}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_JSON)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"Dataset JSON saved successfully to {OUTPUT_JSON}!")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("=== Summary Statistics ===")
    print("=" * 80)
    print(f"Total image-mask pairs: {len(all_pairs)}")
    
    # Count by space
    native_count = sum(1 for p in all_pairs if p['space'] == 'native')
    atlas_count = sum(1 for p in all_pairs if p['space'] == 'atlas')
    print(f"\nBy Space Type:")
    print(f"  Native space pairs: {native_count}")
    print(f"  Atlas space pairs: {atlas_count}")
    
    # Count by contrast weighting
    contrast_counts = {}
    for pair in all_pairs:
        contrast = pair.get('mr_contrast_weighting', 'Unknown')
        contrast_counts[contrast] = contrast_counts.get(contrast, 0) + 1
    
    print(f"\nBy MR Contrast Weighting:")
    for contrast, count in sorted(contrast_counts.items()):
        print(f"  {contrast}: {count}")
    
    # Count by acquisition plane
    plane_counts = {}
    for pair in all_pairs:
        plane = pair.get('acquisition_plane', 'Unknown')
        plane_counts[plane] = plane_counts.get(plane, 0) + 1
    
    print(f"\nBy Acquisition Plane:")
    for plane, count in sorted(plane_counts.items()):
        print(f"  {plane}: {count}")
    
    # Count by field strength
    field_counts = {}
    for pair in all_pairs:
        field = pair.get('field_strength', 'Unknown')
        field_counts[field] = field_counts.get(field, 0) + 1
    
    print(f"\nBy Field Strength:")
    for field, count in sorted(field_counts.items()):
        print(f"  {field}T: {count}")
    
    # Count unique subjects
    unique_subjects = set(p['subject_id'] for p in all_pairs)
    print(f"\nUnique subjects: {len(unique_subjects)}")
    
    # Show first pair as example
    if all_pairs:
        print(f"\n" + "=" * 80)
        print("Example pair (first entry):")
        print("=" * 80)
        print(json.dumps(all_pairs[0], indent=2))
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    generate_dataset_json()
