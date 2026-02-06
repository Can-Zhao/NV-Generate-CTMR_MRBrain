#!/usr/bin/env python3
"""
Script to create JSON for MR embeddings.

This script:
1. Scans embedding directory for all *_emb.nii.gz files
2. Matches embeddings to original images from the brain_mask pairs JSON
3. Extracts target size from filenames
4. Computes new spacing from original spacing and size ratios
5. Maps MR contrast types to class names
6. Splits data by subject_id into training (90%) and testing (10%)
7. Saves to dataset_MR-RATE_emb.json
"""

import os
import json
import glob
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configuration
INPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_MRbrain/jsons/dataset_MR-RATE_brain_mask_pairs.json"
OUTPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_MRbrain/jsons/dataset_MR-RATE_0_emb.json"
EMBEDDING_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/MR-Rate/nvidia_1000_mri_skull_stripped"
SKULL_STRIPPED_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/hyperlinks_removed/datasets--Forithmus--MR-RATE/snapshots/6c419668310c03d150b7904821a5b41ed1123318/nvidia_1000_mri_skull_stripped"

# Test split ratio
TEST_SPLIT_RATIO = 0.2


def get_mri_class(contrast_weighting, contrast_agent_state):
    """
    Map MR contrast type and contrast agent state to class name.
    
    Examples:
        T1 + NON_CONTRAST -> "mri_t1"
        T1 + CONTRAST -> "mri_t1ce"
        FLAIR + NON_CONTRAST -> "mri_flair"
        MRA -> "mri_mra"
    """
    contrast_weighting = (contrast_weighting or "").upper()
    contrast_agent_state = (contrast_agent_state or "").upper()
    
    # Check if MRA (Magnetic Resonance Angiography)
    if "MRA" in contrast_weighting or "ANGIO" in contrast_weighting:
        return "mri_mra"
    
    # Check if contrast enhanced
    is_contrast_enhanced = "CONTRAST" in contrast_agent_state and "NON" not in contrast_agent_state
    
    # Map to class name
    if "T1" in contrast_weighting:
        return "mri_t1c" if is_contrast_enhanced else "mri_t1"
    elif "T2" in contrast_weighting:
        return "mri_t2c" if is_contrast_enhanced else "mri_t2"
    elif "FLAIR" in contrast_weighting:
        return "mri_flairc" if is_contrast_enhanced else "mri_flair"
    else:
        # Default to mri for unknown types
        return "mri"


def get_train_test_split(all_subject_ids, test_ratio=0.1):
    """
    Split subject IDs into training and testing sets.
    
    Args:
        all_subject_ids: List of all subject IDs
        test_ratio: Fraction of subjects for testing (default 0.1 = 10%)
        
    Returns:
        Tuple of (train_subjects_set, test_subjects_set)
    """
    # Sort for deterministic split
    sorted_subjects = sorted(all_subject_ids)
    
    # Calculate split point
    n_test = int(len(sorted_subjects) * test_ratio)
    
    # Last 10% for testing
    test_subjects = set(sorted_subjects[-n_test:])
    train_subjects = set(sorted_subjects[:-n_test])
    
    return train_subjects, test_subjects


def extract_size_from_filename(filename):
    """
    Extract target size from embedding filename.
    
    Example: "native__flair-raw-axi-nc_256x256x128_emb.nii.gz" -> [256, 256, 128]
    
    Args:
        filename: Embedding filename
        
    Returns:
        List of [x, y, z] dimensions, or None if parsing fails
    """
    try:
        # Remove .nii.gz extension
        basename = filename.replace('.nii.gz', '')
        # Split by underscore and find the size part (e.g., "256x256x128")
        parts = basename.split('_')
        
        # Find the part that looks like "XxYxZ" before "emb"
        for i, part in enumerate(parts):
            if part == 'emb' and i > 0:
                size_str = parts[i - 1]
                if 'x' in size_str:
                    dims = [int(d) for d in size_str.split('x')]
                    if len(dims) == 3:
                        return dims
        return None
    except:
        return None


def get_original_image_info(original_image_path, original_json_data):
    """
    Get original image size and spacing from the JSON and skull-stripped image.
    
    Args:
        original_image_path: Relative path to original image
        original_json_data: Dictionary mapping image paths to metadata
        
    Returns:
        Tuple of (original_size, original_spacing, metadata_dict)
    """
    # Get metadata from JSON
    metadata = original_json_data.get(original_image_path)
    if not metadata:
        return None, None, None
    
    # Load skull-stripped image to get size and spacing
    full_path = os.path.join(SKULL_STRIPPED_ROOT, original_image_path)
    if not os.path.exists(full_path):
        return None, None, metadata
    
    try:
        img = nib.load(full_path)
        original_size = list(img.shape)
        
        # Get spacing
        if hasattr(img, 'header') and hasattr(img.header, 'get_zooms'):
            zooms = img.header.get_zooms()
            if len(zooms) >= 3:
                original_spacing = [float(zooms[0]), float(zooms[1]), float(zooms[2])]
            else:
                original_spacing = [1.0, 1.0, 1.0]
        else:
            original_spacing = [1.0, 1.0, 1.0]
        
        return original_size, original_spacing, metadata
    except Exception as e:
        print(f"Warning: Could not load {full_path}: {e}")
        return None, None, metadata


def compute_new_spacing(original_size, original_spacing, new_size):
    """
    Compute new spacing after resampling.
    
    new_spacing[i] = original_spacing[i] * (original_size[i] / new_size[i])
    
    Args:
        original_size: [x, y, z] original dimensions
        original_spacing: [x, y, z] original spacing in mm
        new_size: [x, y, z] new dimensions
        
    Returns:
        List of [x, y, z] new spacing in mm
    """
    new_spacing = [
        original_spacing[i] * (original_size[i] / new_size[i])
        for i in range(3)
    ]
    return new_spacing


def check_spacing_constraint(target_size, new_spacing):
    """
    Check if spacing constraint is satisfied: x-spacing <= z-spacing.
    
    For dimensions in format [x, x, z] or [x, z, x] or [z, x, x],
    where two dimensions are equal (x) and one is different (z),
    we require that the spacing of x dimensions <= spacing of z dimension.
    
    Args:
        target_size: [dim0, dim1, dim2] dimensions
        new_spacing: [sp0, sp1, sp2] spacing values
        
    Returns:
        True if constraint is satisfied, False otherwise
    """
    from collections import Counter
    
    # Count occurrences of each dimension size
    size_counts = Counter(target_size)
    
    # If all three dimensions are equal, we can't determine x vs z
    # Keep these embeddings
    if len(size_counts) == 1:
        return True
    
    # Find which dimension appears most frequently (x) and which is singleton (z)
    if len(size_counts) == 2:
        # Two dimensions are equal (x), one is different (z)
        x_size = None
        z_size = None
        
        for size, count in size_counts.items():
            if count == 2:
                x_size = size
            elif count == 1:
                z_size = size
        
        if x_size is not None and z_size is not None:
            # Get spacing for x and z dimensions
            x_spacings = []
            z_spacing = None
            
            for i in range(3):
                if target_size[i] == x_size:
                    x_spacings.append(new_spacing[i])
                elif target_size[i] == z_size:
                    z_spacing = new_spacing[i]
            
            # Check constraint: x_spacing <= z_spacing
            if len(x_spacings) > 0 and z_spacing is not None:
                avg_x_spacing = sum(x_spacings) / len(x_spacings)
                return avg_x_spacing <= z_spacing
    
    # If all dimensions are different, keep the embedding
    return True


def main():
    print("=" * 80)
    print("Creating MR Embedding JSON")
    print("=" * 80)
    
    # Load original JSON
    print(f"\nLoading original JSON: {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        original_pairs = json.load(f)
    
    print(f"Found {len(original_pairs)} image-mask pairs")
    
    # Create mapping from image path to metadata
    original_json_data = {}
    for pair in original_pairs:
        original_json_data[pair['image']] = pair
    
    # Get unique images
    unique_images = list(set(pair['image'] for pair in original_pairs))
    print(f"Found {len(unique_images)} unique images")
    
    # Scan embedding directory for all *_emb.nii.gz files
    print(f"\nScanning embedding directory: {EMBEDDING_ROOT}")
    embedding_pattern = os.path.join(EMBEDDING_ROOT, "**/*_emb.nii.gz")
    embedding_files = glob.glob(embedding_pattern, recursive=True)
    print(f"Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("ERROR: No embedding files found!")
        return
    
    # First pass: collect all subject IDs
    print("\nCollecting subject IDs...")
    all_subjects = set()
    for pair in original_pairs:
        subject_id = pair.get('subject_id')
        if subject_id:
            all_subjects.add(subject_id)
    
    print(f"Found {len(all_subjects)} unique subjects")
    
    # Split subjects into train/test
    train_subjects, test_subjects = get_train_test_split(list(all_subjects), TEST_SPLIT_RATIO)
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Testing subjects: {len(test_subjects)}")
    
    # Process each embedding file
    print("\nProcessing embeddings...")
    training_data = []
    testing_data = []
    
    processed_count = 0
    skipped_count = 0
    skip_reasons = defaultdict(int)
    
    for emb_file in tqdm(embedding_files, desc="Processing embeddings", unit="file"):
        # Get relative path from embedding root
        emb_rel_path = os.path.relpath(emb_file, EMBEDDING_ROOT)
        
        # Extract target size from filename
        target_size = extract_size_from_filename(os.path.basename(emb_file))
        if target_size is None:
            skip_reasons["no_size_in_filename"] += 1
            skipped_count += 1
            continue
        
        # Find corresponding original image
        # Remove the size suffix and _emb to get base name
        basename = os.path.basename(emb_file)
        # Extract original filename pattern
        # e.g., "native__flair-raw-axi-nc_256x256x128_emb.nii.gz" -> "native__flair-raw-axi-nc.nii.gz"
        
        # First remove .nii.gz extension
        basename_no_ext = basename.replace('.nii.gz', '')
        parts = basename_no_ext.split('_')
        
        # Find index of 'emb'
        try:
            emb_idx = parts.index('emb')
            # Remove size part and 'emb' part
            original_basename = '_'.join(parts[:emb_idx-1]) + '.nii.gz'
        except ValueError:
            skip_reasons["no_emb_in_filename"] += 1
            skipped_count += 1
            continue
        
        # Reconstruct original image path
        # The embedding path structure should match original structure
        emb_dir = os.path.dirname(emb_rel_path)
        original_image_path = os.path.join(emb_dir, original_basename)
        
        # Get original image info
        original_size, original_spacing, metadata = get_original_image_info(
            original_image_path, original_json_data
        )
        
        if original_size is None or original_spacing is None:
            skip_reasons["no_original_data"] += 1
            skipped_count += 1
            continue
        
        # Compute new spacing
        new_spacing = compute_new_spacing(original_size, original_spacing, target_size)
        
        # Check spacing constraint: x-spacing <= z-spacing
        if not check_spacing_constraint(target_size, new_spacing):
            skip_reasons["spacing_constraint_failed"] += 1
            skipped_count += 1
            continue
        
        # Determine MRI class
        mri_class = get_mri_class(
            metadata.get('mr_contrast_weighting'),
            metadata.get('contrast_agent_state')
        )
        
        # Create entry
        entry = {
            "image": emb_rel_path,
            "dim": target_size,
            "spacing": [round(s, 6) for s in new_spacing],  # Round to 6 decimal places
            "class": mri_class,
        }
        
        # Add important metadata fields
        for key in ['subject_id', 'space', 'modality_key', 'mr_contrast_weighting', 
                    'contrast_agent_state', 'acquisition_plane', 'scanner_vendor',
                    'scanner_model', 'field_strength', 'anatomy_region', 'series_role']:
            if key in metadata:
                entry[key] = metadata[key]
        
        # Add original image reference
        entry['original_image'] = original_image_path
        
        # Determine split based on subject_id
        subject_id = metadata.get('subject_id', 'unknown')
        
        if subject_id in test_subjects:
            testing_data.append(entry)
        else:
            training_data.append(entry)
        
        processed_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total embeddings found: {len(embedding_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Training samples: {len(training_data)}")
    print(f"Testing samples: {len(testing_data)}")
    
    if skip_reasons:
        print("\nSkip reasons:")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count}")
    
    # Count by class
    train_classes = defaultdict(int)
    test_classes = defaultdict(int)
    for entry in training_data:
        train_classes[entry['class']] += 1
    for entry in testing_data:
        test_classes[entry['class']] += 1
    
    print("\nTraining samples by class:")
    for cls, count in sorted(train_classes.items()):
        print(f"  {cls}: {count}")
    
    print("\nTesting samples by class:")
    for cls, count in sorted(test_classes.items()):
        print(f"  {cls}: {count}")
    
    # Get unique subjects in each split
    train_subjects = set(e['subject_id'] for e in training_data if 'subject_id' in e)
    test_subjects = set(e['subject_id'] for e in testing_data if 'subject_id' in e)
    
    print(f"\nUnique subjects in training: {len(train_subjects)}")
    print(f"Unique subjects in testing: {len(test_subjects)}")
    if len(train_subjects) + len(test_subjects) > 0:
        print(f"Test ratio: {len(test_subjects) / (len(train_subjects) + len(test_subjects)):.1%}")
    
    # Save to JSON
    output_data = {
        "training": training_data,
        "testing": testing_data
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to: {OUTPUT_JSON}")
    print(f"  Total entries: {len(training_data) + len(testing_data)}")
    
    # Show example entries
    if training_data:
        print("\nExample training entry:")
        example = training_data[0].copy()
        # Show only key fields for brevity
        example_brief = {
            "image": example['image'],
            "dim": example['dim'],
            "spacing": example['spacing'],
            "class": example['class'],
            "subject_id": example.get('subject_id', 'N/A')
        }
        print(json.dumps(example_brief, indent=2))
    
    if testing_data:
        print("\nExample testing entry:")
        example = testing_data[0].copy()
        example_brief = {
            "image": example['image'],
            "dim": example['dim'],
            "spacing": example['spacing'],
            "class": example['class'],
            "subject_id": example.get('subject_id', 'N/A')
        }
        print(json.dumps(example_brief, indent=2))
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
