#!/usr/bin/env python3
"""
Script to generate image-brain_mask pair JSON for MR-RATE dataset.

This script processes the MR-RATE dataset and creates a JSON file containing
image/brain_mask pairs with associated metadata for each subject.

New data structure (20260210):
- Data organized in batches: mri/batch00/, mri/batch01/, etc.
- Each subject has img/ and seg/ directories directly
- Images: {subject_id}_{modality}-raw-{plane}.nii.gz
- Brain masks: {subject_id}_{modality}-raw-{plane}_brain-mask.nii.gz
- Metadata in CSV files: metadata/batch00_metadata.csv

Output: ./jsons/dataset_MR-RATE_brain_mask_pairs.json
"""

import os
import json
import glob
import csv
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri/"
METADATA_ROOT = None
OUTPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_brain_mask_pairs.json"


def extract_modality_key(filename, subject_id):
    """
    Extract modality key from filename, and split into modality and acquisition_plane.
    
    Examples:
        '3226090_flair-raw-axi.nii.gz' -> ('flair-raw-axi', 'flair', 'axi')
        '3226090_t1w-raw-cor.nii.gz' -> ('t1w-raw-cor', 't1w', 'cor')
        '3226090_flair-raw-sag_2.nii.gz' -> ('flair-raw-sag', 'flair', 'sag')  # _2 suffix removed
    
    Args:
        filename (str): Filename to extract modality key from
        subject_id (str): Subject ID to remove from filename
        
    Returns:
        tuple: (modality_key, modality, acquisition_plane)
            modality_key: Full key like 'flair-raw-axi' (without numeric suffix)
            modality: Base modality like 'flair', 't1w', 't2w', 'swi', 'mra'
            acquisition_plane: Plane like 'axi', 'sag', 'cor', 'obl' or None
    """
    import re
    
    basename = os.path.basename(filename)
    basename = basename.replace('.nii.gz', '')
    
    # Remove subject_id prefix if present
    if basename.startswith(f'{subject_id}_'):
        basename = basename[len(subject_id) + 1:]  # Remove '{subject_id}_'
    
    # Remove _brain-mask suffix if present
    if basename.endswith('_brain-mask'):
        basename = basename[:-11]  # Remove '_brain-mask'
    
    # Remove numeric suffixes like _2, _3, _10, -2, -3, -10, etc.
    basename = re.sub(r'[-_]\d+$', '', basename)
    
    # Extract modality and acquisition plane
    modality = None
    acquisition_plane = None
    
    # Try to extract plane (axi, sag, cor, obl)
    for plane in ['axi', 'sag', 'cor', 'obl']:
        if f'-{plane}' in basename:
            acquisition_plane = plane
            # Extract modality (everything before the plane)
            parts = basename.split(f'-{plane}')
            modality_part = parts[0]
            # Extract base modality (e.g., 'flair-raw' -> 'flair', 't1w-raw' -> 't1w')
            if '-raw' in modality_part:
                modality = modality_part.split('-raw')[0]
            else:
                # Fallback: use first part before any dash
                modality = modality_part.split('-')[0]
            break
    
    # If no plane found, try to extract modality from common patterns
    if modality is None:
        # Common modalities: flair, t1w, t2w, swi, mra
        for mod in ['flair', 't1w', 't2w', 'swi', 'mra']:
            if basename.startswith(mod):
                modality = mod
                break
    
    return basename, modality, acquisition_plane


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


def load_metadata_from_csv(metadata_root):
    """
    Load metadata from CSV files in metadata directory.
    
    Args:
        metadata_root (str): Root directory containing metadata CSV files
        
    Returns:
        dict: Dictionary mapping (patient_id, modality_id) -> metadata dict
    """
    metadata_dict = {}
    
    # Find all metadata CSV files
    csv_files = glob.glob(os.path.join(metadata_root, "batch*_metadata.csv"))
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    patient_id = row.get('patient_id', '')
                    modality_id = row.get('modality_id', '')
                    key = (patient_id, modality_id)
                    
                    # Store all metadata fields
                    metadata_dict[key] = dict(row)
        except Exception as e:
            print(f"Warning: Failed to load metadata from {csv_file}: {e}")
    
    return metadata_dict


def process_subject(subject_dir, subject_id, metadata_dict, data_root):
    """
    Process subject: 1-to-1 matching between images and brain_masks.
    
    For each image in img/, find the corresponding brain_mask in seg/
    by adding '_brain-mask' suffix.
    
    Args:
        subject_dir (str): Path to subject directory
        subject_id (str): Subject ID
        metadata_dict (dict): Metadata dictionary mapping (patient_id, modality_id) -> metadata
        data_root (str): Root directory of the dataset
        
    Returns:
        list: List of data pair dictionaries
    """
    pairs = []
    
    img_dir = os.path.join(subject_dir, "img")
    seg_dir = os.path.join(subject_dir, "seg")
    
    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(seg_dir):
        return pairs
    
    # Get all .nii.gz files in img directory (excluding brain masks)
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    # Filter out brain masks and defacing masks
    image_files = [f for f in image_files if '_brain-mask' not in os.path.basename(f) 
                   and '_defacing-mask' not in os.path.basename(f)]
    
    for image_path in image_files:
        # Construct corresponding brain_mask path
        image_basename = os.path.basename(image_path)
        # Replace .nii.gz with _brain-mask.nii.gz
        mask_basename = image_basename.replace('.nii.gz', '_brain-mask.nii.gz')
        mask_path = os.path.join(seg_dir, mask_basename)
        
        # Check if brain_mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Brain mask not found for {image_path}")
            continue
        
        # Extract modality key, modality, and acquisition plane
        modality_key, modality, acquisition_plane = extract_modality_key(image_basename, subject_id)
        
        # Print warning if acquisition_plane is None
        if acquisition_plane is None:
            print(f"Warning: Could not extract acquisition_plane from {image_basename} (subject: {subject_id}, modality_key: {modality_key})")
        
        # Get metadata for this subject and modality
        # Try to match with metadata using patient_id and modality_id
        metadata = {}
        for (patient_id, modality_id), meta in metadata_dict.items():
            if patient_id == subject_id and modality_id == modality_key:
                metadata = meta
                break
        
        # Convert to relative paths
        image_rel_path = get_relative_path(image_path, data_root)
        mask_rel_path = get_relative_path(mask_path, data_root)
        
        # Create data pair entry
        pair = {
            "image": image_rel_path,
            "brain_mask": mask_rel_path,
            "subject_id": subject_id,
            "modality_key": modality_key,
        }
        
        # Add modality and acquisition_plane if extracted
        if modality:
            pair["modality"] = modality
        if acquisition_plane:
            pair["acquisition_plane"] = acquisition_plane
        
        # Add all metadata fields
        pair.update(metadata)
        
        pairs.append(pair)
    
    return pairs




def generate_dataset_json():
    """
    Main function to generate dataset JSON file.
    
    Processes all subjects in the MR-RATE dataset and creates a JSON file
    containing image/brain_mask pairs with metadata.
    """
    all_pairs = []
    
    # Load metadata from CSV files (skip if METADATA_ROOT is not available)
    if METADATA_ROOT and os.path.isdir(METADATA_ROOT):
        print("Loading metadata from CSV files...")
        metadata_dict = load_metadata_from_csv(METADATA_ROOT)
        print(f"Loaded metadata for {len(metadata_dict)} entries")
    else:
        print("METADATA_ROOT is not set or does not exist, skipping metadata loading.")
        metadata_dict = {}
    print("=" * 80)
    
    # Get all batch directories
    batch_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "batch*")))
    batch_dirs = [d for d in batch_dirs if os.path.isdir(d)]
    
    print(f"Found {len(batch_dirs)} batches in {DATA_ROOT}")
    
    # Process each batch
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        print(f"\nProcessing batch: {batch_name}")
        
        # Get all subject directories in this batch
        subject_dirs = sorted(glob.glob(os.path.join(batch_dir, "*")))
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
        
        print(f"  Found {len(subject_dirs)} subjects in {batch_name}")
        
        # Process each subject
        for idx, subject_dir in enumerate(subject_dirs):
            subject_id = os.path.basename(subject_dir)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processing subject {idx + 1}/{len(subject_dirs)}: {subject_id}")
            
            try:
                # Process subject (img/ and seg/ directories)
                pairs = process_subject(subject_dir, subject_id, metadata_dict, DATA_ROOT)
                all_pairs.extend(pairs)
                
            except Exception as e:
                print(f"  Error processing subject {subject_id}: {e}")
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
    
    # Count by modality
    modality_counts = {}
    for pair in all_pairs:
        modality = pair.get('modality_key', 'Unknown')
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    print(f"\nBy Modality:")
    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality}: {count}")
    
    # Count by classified modality (from metadata)
    classified_counts = {}
    for pair in all_pairs:
        classified = pair.get('classified_modality', 'Unknown')
        classified_counts[classified] = classified_counts.get(classified, 0) + 1
    
    print(f"\nBy Classified Modality:")
    for classified, count in sorted(classified_counts.items()):
        print(f"  {classified}: {count}")
    
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
        field = pair.get('FieldStrength_T', 'Unknown')
        if field:
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
