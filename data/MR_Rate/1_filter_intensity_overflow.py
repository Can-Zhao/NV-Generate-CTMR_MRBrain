#!/usr/bin/env python3
"""
Filter out MR-RATE entries with uint16→int16 overflow artifacts.

Overflow detection:
- Reads raw voxel integers (no float conversion, no slope/intercept scaling)
- Filters images where min < -100 (indicates uint16→int16 wrap-around)

A minimum below -100 indicates a uint16→int16 overflow artifact (wrap-around),
where background voxels that were stored as large unsigned values are
misinterpreted as large negative signed values.

Input:  jsons/dataset_MR-RATE_brain_mask_pairs.json  (35 k entries)
Output: jsons/dataset_MR-RATE_brain_mask_pairs_filtered.json

Usage:
    python 1_filter_intensity_overflow.py [--dry-run] [--workers N]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = (
    "/lustre/fsw/portfolios/healthcareeng/projects/"
    "healthcareeng_monai/datasets/MR-RATE/mrrate_temp_1/mri/"
)
INPUT_JSON = (
    "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/"
    "dataset_MR-RATE_brain_mask_pairs.json"
)
OUTPUT_JSON = (
    "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/"
    "dataset_MR-RATE_brain_mask_pairs_filtered.json"
)
MIN_THRESHOLD = -100  # images with min < this value are filtered out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def check_entry(entry: dict) -> tuple[bool, dict, str]:
    """
    Load the image and check whether its minimum intensity is acceptable.

    Args:
        entry: One JSON entry with at least an ``image`` relative path.

    Returns:
        (keep, entry, message)
            keep    – True if the image passes the threshold.
            entry   – The original entry dict (passed through for convenience).
            message – Human-readable description of the decision.
    """
    rel_path = entry["image"]
    full_path = os.path.join(DATA_ROOT, rel_path)

    if not os.path.exists(full_path):
        return False, entry, f"MISSING  {rel_path}"

    try:
        # mmap=False is critical on Lustre/NFS: memory-mapped I/O can deadlock
        img = nib.load(full_path, mmap=False)
        header_dtype = str(img.header.get_data_dtype())
        # Read raw voxel integers – no slope/intercept scaling applied
        # This is critical: get_fdata() applies scaling which can mask overflow
        raw = np.array(img.dataobj)
        min_val = float(raw.min())
        max_val = float(raw.max())
    except Exception as exc:
        return False, entry, f"ERROR    {rel_path}: {exc}"

    # Check for overflow: negative values in int16 indicate uint16→int16 wrap-around
    # When uint16 values > 32767 are stored as int16, they wrap to negative values
    has_overflow = min_val < MIN_THRESHOLD
    
    if has_overflow:
        return False, entry, f"FILTERED {rel_path}  (min={min_val:.1f}, max={max_val:.1f}, dtype={header_dtype})"

    return True, entry, f"OK       {rel_path}  (min={min_val:.1f}, max={max_val:.1f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only; do not write the output JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel worker processes (default: 32).",
    )
    args = parser.parse_args()

    # Load input JSON
    print(f"Loading {INPUT_JSON} …")
    with open(INPUT_JSON) as f:
        entries = json.load(f)
    print(f"  Total entries: {len(entries)}")

    # Check each image in parallel
    kept = []
    filtered = []
    errors = []

    print(f"\nChecking minimum intensities with {args.workers} workers …")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_entry, e): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(entries), unit="img"):
            keep, entry, message = future.result()
            if keep:
                kept.append(entry)
            elif message.startswith("ERROR") or message.startswith("MISSING"):
                errors.append(message)
                filtered.append(entry)
            else:
                filtered.append(entry)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Kept:     {len(kept):>6}")
    print(f"  Filtered: {len(filtered):>6}  (min < {MIN_THRESHOLD})")
    print(f"  Errors:   {len(errors):>6}  (missing / unreadable)")
    print(f"{'='*60}")

    if errors:
        print("\nErrors / missing files:")
        for msg in errors:
            print(f"  {msg}")

    if filtered:
        print(f"\nFiltered entries (showing first 20):")
        shown = [e for e in filtered if not any(
            e["image"] in msg for msg in errors
        )][:20]
        for e in shown:
            print(f"  {e['image']}")

    # Write output
    if not args.dry_run:
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(kept, f, indent=2)
        print(f"\nSaved {len(kept)} entries → {OUTPUT_JSON}")
    else:
        print("\n[dry-run] Output JSON not written.")


if __name__ == "__main__":
    main()
