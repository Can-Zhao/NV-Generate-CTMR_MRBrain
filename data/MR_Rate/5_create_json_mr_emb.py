#!/usr/bin/env python3
"""
Script to create JSON for MR embeddings.

This script:
1. Scans embedding directory for all *_emb.nii.gz files
2. Matches embeddings to original images from the brain_mask pairs JSON
3. Extracts target size from filenames
4. Computes new spacing from original spacing and size ratios
5. Maps MR contrast types to class names from the embedding filename
   (no contrast-enhanced variants; supported: t1w, t2w, flair, swi, mra).
   If the path contains "skull_stripped" or "skull-stripped", uses
   mri_*_skull_stripped classes from configs/modality_mapping.json;
   otherwise uses whole-brain classes (mri_t1, mri_t2, ...).
6. Splits data by subject_id into training and testing sets
7. Saves to dataset_MR-RATE_emb.json

Embedding filename format:
  {subject_id}_{contrast_type}-raw-{plane}[-{suffix}]_{XxYxZ}_emb.nii.gz
  e.g. 3970624_t2w-raw-axi_256x256x128_emb.nii.gz
"""

import os
import json
import glob
import argparse
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool

# Configuration
INPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_brain_mask_pairs.json"
OUTPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_0_emb.json"
EMBEDDING_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/MR-Rate/MR-RATE_20260227_unzip/mri/"
SKULL_STRIPPED_ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri/"

# Modality mapping: path to config (whole brain vs skull-stripped class names)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODALITY_MAPPING_JSON = os.path.join(_SCRIPT_DIR, "..", "..", "configs", "modality_mapping.json")

# Test split ratio
TEST_SPLIT_RATIO = 0.1

# Parallel: number of workers for processing embeddings (set via --num-workers)
NUM_WORKERS = 96

# Contrast type -> class name (whole brain). Skull-stripped variants use _skull_stripped suffix.
_CONTRAST_TO_WHOLE_BRAIN = {
    "t1w": "mri_t1",
    "t2w": "mri_t2",
    "flair": "mri_flair",
    "swi": "mri_swi",
    "mra": "mri_mra",
}


def _is_skull_stripped_path(embedding_root, embedding_rel_path=""):
    """
    Return True if the data is skull-stripped, based on path.
    If 'skull_stripped' or 'skull-stripped' appears in the embedding root or
    in the relative path, the data is treated as skull stripped.
    """
    combined = (embedding_root + " " + embedding_rel_path).lower()
    return "skull_stripped" in combined or "skull-stripped" in combined


def get_mri_class_from_filename(basename, is_skull_stripped=False):
    """
    Extract MRI class from the embedding filename.

    Filename format: {subject_id}_{contrast_type}-raw-{plane}[-{suffix}]_{size}_emb.nii.gz
    The contrast type is the segment between the first '_' and '-raw-'.

    Uses modality_mapping.json: whole brain (mri_t1, mri_t2, ...) vs
    skull stripped (mri_t1_skull_stripped, mri_t2_skull_stripped, ...).
    If is_skull_stripped is True (e.g. path contains "skull_stripped"), returns
    skull-stripped class names; otherwise whole-brain class names.

    Args:
        basename: Embedding filename (basename only)
        is_skull_stripped: If True, use mri_*_skull_stripped classes.

    Returns:
        MRI class string, or "mri" for unknown types.
    """
    name = basename.replace('.nii.gz', '')
    parts = name.split('_')
    if len(parts) < 2:
        return "mri"

    contrast_segment = parts[1]
    contrast_type = contrast_segment.split('-')[0].lower()

    base_class = _CONTRAST_TO_WHOLE_BRAIN.get(contrast_type)
    if base_class is None:
        return "mri"
    if is_skull_stripped:
        return base_class + "_skull_stripped"
    return base_class


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

    Example: "3970624_t2w-raw-axi_256x256x128_emb.nii.gz" -> [256, 256, 128]

    Args:
        filename: Embedding filename (basename)

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


def get_original_image_info(original_image_path, original_json_data, skull_stripped_root=None):
    """
    Get original image size and spacing from the JSON and skull-stripped image.
    
    Args:
        original_image_path: Relative path to original image
        original_json_data: Dictionary mapping image paths to metadata
        skull_stripped_root: Optional override for SKULL_STRIPPED_ROOT (used by workers).
        
    Returns:
        Tuple of (original_size, original_spacing, metadata_dict)
    """
    root = skull_stripped_root if skull_stripped_root is not None else SKULL_STRIPPED_ROOT
    # Get metadata from JSON
    metadata = original_json_data.get(original_image_path)
    if not metadata:
        return None, None, None
    
    # Load skull-stripped image to get size and spacing
    full_path = os.path.join(root, original_image_path)
    if not os.path.exists(full_path):
        return None, None, metadata
    
    try:
        img = nib.load(full_path)
        # Reorient to RAS (same as MONAI does in 4_apply_vae.py) so that
        # spacing axes match the RAS convention [sp_X, sp_Y, sp_Z] used by
        # the embedding pipeline. Without this, get_zooms() returns spacing
        # in the native storage order, which may differ from RAS order.
        img = nib.as_closest_canonical(img)
        original_size = list(img.shape[:3])
        
        # Get spacing in RAS order after reorientation
        zooms = img.header.get_zooms()
        if len(zooms) >= 3:
            original_spacing = [float(zooms[0]), float(zooms[1]), float(zooms[2])]
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


# ── Parallel worker (uses _worker_ctx set by _init_worker) ─────────────────
_worker_ctx = {}


def _init_worker(orig_json_data, train_subjects, test_subjects, embedding_root, skull_stripped_root):
    """Set globals in worker process so process_one_embedding can use them."""
    _worker_ctx["original_json_data"] = orig_json_data
    _worker_ctx["train_subjects"] = train_subjects
    _worker_ctx["test_subjects"] = test_subjects
    _worker_ctx["embedding_root"] = embedding_root
    _worker_ctx["skull_stripped_root"] = skull_stripped_root


def _process_one_embedding(args):
    """
    Process a single embedding file. Returns (entry, split, skip_reason).
    split is "train" or "test"; skip_reason is non-None if skipped.
    """
    emb_file, emb_rel_path = args
    orig_data = _worker_ctx["original_json_data"]
    train_subs = _worker_ctx["train_subjects"]
    test_subs = _worker_ctx["test_subjects"]
    emb_root = _worker_ctx["embedding_root"]
    skull_root = _worker_ctx["skull_stripped_root"]

    target_size = extract_size_from_filename(os.path.basename(emb_file))
    if target_size is None:
        return None, None, "no_size_in_filename"

    basename = os.path.basename(emb_file)
    basename_no_ext = basename.replace('.nii.gz', '')
    parts = basename_no_ext.split('_')
    try:
        emb_idx = parts.index('emb')
        if emb_idx < 2:
            return None, None, "invalid_filename_format"
        original_basename = '_'.join(parts[: emb_idx - 1]) + '.nii.gz'
    except ValueError:
        return None, None, "no_emb_in_filename"

    emb_dir = os.path.dirname(emb_rel_path)
    original_image_path = os.path.join(emb_dir, original_basename)

    original_size, original_spacing, metadata = get_original_image_info(
        original_image_path, orig_data, skull_stripped_root=skull_root
    )
    if original_size is None or original_spacing is None:
        return None, None, "no_original_data"

    new_spacing = compute_new_spacing(original_size, original_spacing, target_size)
    is_skull_stripped = _is_skull_stripped_path(emb_root, emb_rel_path)
    mri_class = get_mri_class_from_filename(basename, is_skull_stripped=is_skull_stripped)

    entry = {
        "image": emb_rel_path,
        "dim": target_size,
        "spacing": [round(s, 6) for s in new_spacing],
        "class": mri_class,
    }
    for key in ['subject_id', 'modality_key', 'modality', 'acquisition_plane']:
        if key in metadata:
            entry[key] = metadata[key]
    entry["original_image"] = original_image_path

    subject_id = metadata.get('subject_id', 'unknown')
    split = "test" if subject_id in test_subs else "train"
    return entry, split, None


def _load_modality_mapping():
    """Load modality_mapping.json; return dict of class_name -> id. Used to validate class names."""
    if not os.path.isfile(MODALITY_MAPPING_JSON):
        return {}
    with open(MODALITY_MAPPING_JSON, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Create MR embedding JSON from embedding files and brain_mask pairs.")
    parser.add_argument(
        "--num-workers", "-n",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel workers (default: {NUM_WORKERS})",
    )
    args = parser.parse_args()
    num_workers = max(1, args.num_workers)

    print("=" * 80)
    print("Creating MR Embedding JSON")
    print("=" * 80)
    
    # Optional: validate that our class names exist in configs/modality_mapping.json
    modality_ids = _load_modality_mapping()
    if modality_ids:
        for name in list(_CONTRAST_TO_WHOLE_BRAIN.values()) + [
            c + "_skull_stripped" for c in _CONTRAST_TO_WHOLE_BRAIN.values()
        ]:
            if name not in modality_ids:
                print(f"Warning: class '{name}' not in {MODALITY_MAPPING_JSON}")
    
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
    
    # Build list of (emb_file, emb_rel_path) for parallel processing
    task_list = [
        (emb_file, os.path.relpath(emb_file, EMBEDDING_ROOT))
        for emb_file in embedding_files
    ]
    print(f"\nProcessing {len(task_list)} embeddings with {num_workers} workers...")
    training_data = []
    testing_data = []
    skip_reasons = defaultdict(int)
    processed_count = 0
    skipped_count = 0

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            original_json_data,
            train_subjects,
            test_subjects,
            EMBEDDING_ROOT,
            SKULL_STRIPPED_ROOT,
        ),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one_embedding, task_list, chunksize=min(500, max(1, len(task_list) // (num_workers * 4)))),
            total=len(task_list),
            desc="Processing embeddings",
            unit="file",
        ):
            entry, split, skip_reason = result
            if skip_reason is not None:
                skip_reasons[skip_reason] += 1
                skipped_count += 1
                continue
            if split == "test":
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
