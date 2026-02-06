#!/usr/bin/env python3
"""
Script to create JSON for MR embeddings (generic datasets).

This script:
1. Scans embedding directory for all *_emb.nii.gz files
2. Matches embeddings to original images from input JSON
3. Extracts target size from filenames
4. Computes new spacing from original spacing and size ratios
5. Infers MRI class from filename
6. Preserves training/testing split from original JSON
7. Saves to output JSON

For datasets like aomic, qtim, etc.
"""

import os
import json
import glob
import argparse
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def infer_mri_class_from_filename(filename):
    """
    Infer MRI class from filename patterns.
    
    Examples:
        "sub-0001_run-1_T1w.nii.gz" -> "mri_t1"
        "sub-0001_T2w.nii.gz" -> "mri_t2"
        "sub-0001_FLAIR.nii.gz" -> "mri_flair"
        "sub-0001_T1ce.nii.gz" -> "mri_t1c"
    
    Args:
        filename: Image filename
        
    Returns:
        Class string (mri_t1, mri_t2, mri_flair, mri_t1c, mri_t2c, mri_flairc, or mri)
    """
    filename_upper = filename.upper()
    
    # Check for contrast-enhanced first (more specific)
    if 'T1CE' in filename_upper or 'T1_CE' in filename_upper or 'T1-CE' in filename_upper:
        return "mri_t1c"
    elif 'T2CE' in filename_upper or 'T2_CE' in filename_upper or 'T2-CE' in filename_upper:
        return "mri_t2c"
    elif 'FLAIRCE' in filename_upper or 'FLAIR_CE' in filename_upper or 'FLAIR-CE' in filename_upper:
        return "mri_flairc"
    # Then check for non-contrast
    elif 'T1W' in filename_upper or '_T1.' in filename_upper or '_T1_' in filename_upper:
        return "mri_t1"
    elif 'T2W' in filename_upper or '_T2.' in filename_upper or '_T2_' in filename_upper:
        return "mri_t2"
    elif 'FLAIR' in filename_upper:
        return "mri_flair"
    else:
        return "mri"  # Default


def extract_size_from_filename(filename):
    """
    Extract target size from embedding filename.
    
    Example: "sub-0001_run-1_T1w_256x256x128_emb.nii.gz" -> [256, 256, 128]
    
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


def get_original_image_info(original_image_path, input_root):
    """
    Get original image size and spacing.
    
    Args:
        original_image_path: Relative path to original image
        input_root: Root directory for input images
        
    Returns:
        Tuple of (original_size, original_spacing) or (None, None) if error
    """
    full_path = os.path.join(input_root, original_image_path)
    if not os.path.exists(full_path):
        return None, None
    
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
        
        return original_size, original_spacing
    except Exception as e:
        print(f"Warning: Could not load {full_path}: {e}")
        return None, None


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


def main(dataset_name, input_json, input_root, embedding_root, output_json):
    """
    Main function to create embedding JSON.
    
    Args:
        dataset_name: Name of dataset
        input_json: Path to original JSON file
        input_root: Root directory for original images
        embedding_root: Root directory for embeddings
        output_json: Path to output JSON file
    """
    print("=" * 80)
    print(f"Creating MR Embedding JSON for {dataset_name}")
    print("=" * 80)
    
    # Load original JSON
    print(f"\nLoading original JSON: {input_json}")
    with open(input_json, 'r') as f:
        original_data = json.load(f)
    
    # Build mapping: image_path -> split (training/testing)
    image_to_split = {}
    for item in original_data.get('training', []):
        image_to_split[item['image']] = 'training'
    # Support both 'testing' and 'validation' keys
    for item in original_data.get('testing', []):
        image_to_split[item['image']] = 'testing'
    for item in original_data.get('validation', []):
        image_to_split[item['image']] = 'testing'  # Map validation to testing
    
    print(f"Training images: {len([s for s in image_to_split.values() if s == 'training'])}")
    print(f"Testing images: {len([s for s in image_to_split.values() if s == 'testing'])}")
    
    # Scan embedding directory
    print(f"\nScanning embedding directory: {embedding_root}")
    embedding_pattern = os.path.join(embedding_root, "**/*_emb.nii.gz")
    embedding_files = glob.glob(embedding_pattern, recursive=True)
    print(f"Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        print("ERROR: No embedding files found!")
        return
    
    # Process embeddings
    print("\nProcessing embeddings...")
    training_data = []
    testing_data = []
    
    processed_count = 0
    skipped_count = 0
    skip_reasons = defaultdict(int)
    
    for emb_file in tqdm(embedding_files, desc="Processing embeddings", unit="file"):
        # Get relative path from embedding root
        emb_rel_path = os.path.relpath(emb_file, embedding_root)
        
        # Extract target size from filename
        target_size = extract_size_from_filename(os.path.basename(emb_file))
        if target_size is None:
            skip_reasons["no_size_in_filename"] += 1
            skipped_count += 1
            continue
        
        # Find corresponding original image
        # Remove the size suffix and _emb to get base name
        basename = os.path.basename(emb_file)
        # Remove .nii.gz, then split and remove size and emb parts
        basename_no_ext = basename.replace('.nii.gz', '')
        parts = basename_no_ext.split('_')
        
        try:
            emb_idx = parts.index('emb')
            # Remove size part and 'emb' part
            original_basename = '_'.join(parts[:emb_idx-1]) + '.nii.gz'
        except ValueError:
            skip_reasons["no_emb_in_filename"] += 1
            skipped_count += 1
            continue
        
        # Reconstruct original image path
        emb_dir = os.path.dirname(emb_rel_path)
        original_image_path = os.path.join(emb_dir, original_basename)
        
        # Check if this image is in our original JSON
        if original_image_path not in image_to_split:
            skip_reasons["not_in_original_json"] += 1
            skipped_count += 1
            continue
        
        # Get original image info
        original_size, original_spacing = get_original_image_info(original_image_path, input_root)
        
        if original_size is None or original_spacing is None:
            skip_reasons["cannot_load_original"] += 1
            skipped_count += 1
            continue
        
        # Compute new spacing
        new_spacing = compute_new_spacing(original_size, original_spacing, target_size)
        
        # Check spacing constraint: x-spacing <= z-spacing
        if not check_spacing_constraint(target_size, new_spacing):
            skip_reasons["spacing_constraint_failed"] += 1
            skipped_count += 1
            continue
        
        # Infer MRI class from filename
        mri_class = infer_mri_class_from_filename(original_basename)
        
        # Create entry
        entry = {
            "image": emb_rel_path,
            "dim": target_size,
            "spacing": [round(s, 6) for s in new_spacing],
            "class": mri_class,
            "original_image": original_image_path
        }
        
        # Add to appropriate split
        split = image_to_split[original_image_path]
        if split == 'training':
            training_data.append(entry)
        else:
            testing_data.append(entry)
        
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
    
    # Save to JSON
    output_data = {
        "training": training_data,
        "testing": testing_data
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to: {output_json}")
    print(f"  Total entries: {len(training_data) + len(testing_data)}")
    
    # Show example entries
    if training_data:
        print("\nExample training entry:")
        example = training_data[0].copy()
        example_brief = {
            "image": example['image'],
            "dim": example['dim'],
            "spacing": example['spacing'],
            "class": example['class']
        }
        print(json.dumps(example_brief, indent=2))
    
    if testing_data:
        print("\nExample testing entry:")
        example = testing_data[0].copy()
        example_brief = {
            "image": example['image'],
            "dim": example['dim'],
            "spacing": example['spacing'],
            "class": example['class']
        }
        print(json.dumps(example_brief, indent=2))
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create embedding JSON for generic datasets'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name (e.g., "aomic", "qtim")'
    )
    parser.add_argument(
        '--input-json',
        type=str,
        required=True,
        help='Path to original JSON file'
    )
    parser.add_argument(
        '--input-root',
        type=str,
        required=True,
        help='Root directory for original images'
    )
    parser.add_argument(
        '--embedding-root',
        type=str,
        required=True,
        help='Root directory for embeddings'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        required=True,
        help='Path to output JSON file'
    )
    
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset_name,
        input_json=args.input_json,
        input_root=args.input_root,
        embedding_root=args.embedding_root,
        output_json=args.output_json
    )
