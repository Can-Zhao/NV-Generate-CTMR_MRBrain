#!/usr/bin/env python3
"""
Identify and correct MR-RATE images that suffer from a uint16→int16 dtype mismatch.

Root-cause hypothesis
---------------------
The NIfTI header declares the voxel storage dtype as ``int16``, but the scanner
actually wrote ``uint16`` values.  As a result, background voxels encoded as
large unsigned integers (e.g. 65535 = 0xFFFF) are misinterpreted as large
negative signed values (-1), producing a minimum intensity far below -100.

What this script does
---------------------
1. SCAN   – Reads every entry in the input JSON in parallel.
             Images whose minimum intensity < MIN_THRESHOLD are flagged.
2. VERIFY – For flagged images, reports the dtype stored in the NIfTI header
             (expected: ``int16``).  This confirms or refutes the hypothesis.
3. FIX    – (optional, with --fix)  Re-interprets the raw voxel bytes as
             ``uint16``, updates the header dtype accordingly, and writes the
             corrected file to OUTPUT_DIR, preserving the relative-path
             directory structure.

Usage
-----
# List overflow images and inspect their header dtypes (no files written):
    python 1_1_filter_correct_intensity_overflow.py --list-only

# List AND write dtype-corrected copies to the default OUTPUT_DIR:
    python 1_1_filter_correct_intensity_overflow.py --fix

# List AND fix, writing to a custom directory:
    python 1_1_filter_correct_intensity_overflow.py --fix --output-dir /path/to/fixed
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
OVERFLOW_TXT = (
    "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/"
    "dataset_MR-RATE_brain_mask_pairs_overflow.txt"
)
DEFAULT_OUTPUT_DIR = (
    "/lustre/fsw/portfolios/healthcareeng/projects/"
    "healthcareeng_monai/datasets/MR-RATE/mrrate_temp_1_corrected/mri/"
)
MIN_THRESHOLD = -100  # images with min < this value are considered overflow


# ---------------------------------------------------------------------------
# Worker: scan one image
# ---------------------------------------------------------------------------
def scan_entry(entry: dict) -> dict:
    """
    Load a NIfTI image, record its header dtype, minimum and maximum raw value.

    Uses ``np.asanyarray(img.dataobj)`` to read the **raw** stored integers
    (int16) without applying any NIfTI slope/intercept rescaling, so that the
    true signed values are visible.

    Returns a result dict with keys:
        entry, rel_path, full_path, status,
        header_dtype, min_val, max_val, overflow   (last four only on success)
    """
    rel_path = entry["image"]
    full_path = os.path.join(DATA_ROOT, rel_path)
    result = {"entry": entry, "rel_path": rel_path, "full_path": full_path}

    if not os.path.exists(full_path):
        result["status"] = "MISSING"
        return result

    try:
        img = nib.load(full_path)
        # Read raw voxel integers – no float conversion, no slope/intercept
        raw = np.asanyarray(img.dataobj)
        header_dtype = str(img.header.get_data_dtype())
        min_val = float(raw.min())
        max_val = float(raw.max())
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        return result

    result["header_dtype"] = header_dtype
    result["min_val"] = min_val
    result["max_val"] = max_val
    result["overflow"] = min_val < MIN_THRESHOLD
    result["status"] = "OVERFLOW" if result["overflow"] else "OK"
    return result


# ---------------------------------------------------------------------------
# Worker: fix one image
# ---------------------------------------------------------------------------
def fix_entry(result: dict, output_dir: str) -> tuple[bool, str]:
    """
    Re-interpret raw int16 voxel bytes as uint16 and save a corrected copy.

    Strategy
    --------
    The bytes on disk are correct (they are genuine uint16 values); only the
    dtype tag in the NIfTI header is wrong.  We therefore:
      1. Load the raw voxel array as int16 (as the header claims).
      2. Call ``.view(np.uint16)`` – same memory, different dtype interpretation.
      3. Write a new NIfTI with the header dtype patched to uint16.

    NIfTI scaling fields ``scl_slope`` / ``scl_inter`` are cleared (set to 0)
    because the corrected values no longer require affine rescaling.

    Returns
    -------
    (success, message)
    """
    rel_path = result["rel_path"]
    full_path = result["full_path"]
    out_path = os.path.join(output_dir, rel_path)

    try:
        img = nib.load(full_path)
        raw = np.asanyarray(img.dataobj)  # int16 array, shape (X, Y, Z [, T, …])

        # Guard: only fix images whose stored dtype is genuinely int16
        if raw.dtype != np.int16:
            return False, (
                f"SKIP  {rel_path}  "
                f"(stored dtype={raw.dtype}, expected int16 – manual inspection needed)"
            )

        # Reinterpret bytes: int16 0xFFFF → uint16 65535, etc.
        corrected = raw.view(np.uint16)

        # Build a corrected header
        new_header = img.header.copy()
        new_header.set_data_dtype(np.uint16)
        # scl_slope == 0 means "no scaling" per the NIfTI-1 spec
        new_header["scl_slope"] = 0
        new_header["scl_inter"] = 0

        new_img = nib.Nifti1Image(corrected, img.affine, new_header)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(new_img, out_path)

        new_min = float(corrected.min())
        new_max = float(corrected.max())
        return True, (
            f"FIXED {rel_path}  "
            f"(int16 min={result['min_val']:.0f} → uint16 min={new_min:.0f}, "
            f"max={new_max:.0f})"
        )

    except Exception as exc:
        return False, f"FIX_ERR {rel_path}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list-only",
        action="store_true",
        help="Scan and list overflow images with their header dtypes. No files are written.",
    )
    mode.add_argument(
        "--fix",
        action="store_true",
        help="Scan, verify, then write dtype-corrected copies to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for corrected images (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load JSON
    # ------------------------------------------------------------------
    print(f"Loading {INPUT_JSON} …")
    with open(INPUT_JSON) as f:
        entries = json.load(f)
    print(f"  Total entries: {len(entries)}")

    # ------------------------------------------------------------------
    # Step 2: Scan all images in parallel
    # ------------------------------------------------------------------
    print(f"\n[SCAN] Inspecting raw dtype and min/max with {args.workers} workers …")
    scan_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_entry, e): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(entries), unit="img"):
            scan_results.append(future.result())

    ok_results       = [r for r in scan_results if r["status"] == "OK"]
    overflow_results = [r for r in scan_results if r["status"] == "OVERFLOW"]
    missing_results  = [r for r in scan_results if r["status"] == "MISSING"]
    error_results    = [r for r in scan_results if r["status"] == "ERROR"]

    # ------------------------------------------------------------------
    # Step 3: Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  OK:       {len(ok_results):>6}")
    print(f"  OVERFLOW: {len(overflow_results):>6}  (min < {MIN_THRESHOLD})")
    print(f"  MISSING:  {len(missing_results):>6}")
    print(f"  ERRORS:   {len(error_results):>6}")
    print(f"{'='*70}")

    if error_results:
        print("\n[ERRORS]")
        for r in error_results:
            print(f"  {r['rel_path']}: {r.get('error', '')}")

    if missing_results:
        print("\n[MISSING FILES]")
        for r in missing_results:
            print(f"  {r['rel_path']}")

    # ------------------------------------------------------------------
    # Step 4: Verify dtype hypothesis on overflow images
    # ------------------------------------------------------------------
    if not overflow_results:
        print("\nNo overflow images found. Nothing to fix.")
        return

    # Tally dtype tags across all overflow images
    dtype_counts: dict[str, int] = {}
    for r in overflow_results:
        d = r.get("header_dtype", "unknown")
        dtype_counts[d] = dtype_counts.get(d, 0) + 1

    print(f"\n[VERIFY] Header dtype distribution across {len(overflow_results)} overflow images:")
    for dtype_name, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        if dtype_name == "int16":
            tag = "  ✓ confirms uint16→int16 mismatch hypothesis"
        elif dtype_name == "uint16":
            tag = "  ✗ already uint16 – overflow has a different cause"
        else:
            tag = "  ? unexpected dtype"
        print(f"  {dtype_name:<12}  {count:>6} images{tag}")

    # Print full overflow list
    print(f"\n[OVERFLOW LIST]  ({len(overflow_results)} images with min < {MIN_THRESHOLD})")
    print(f"  {'header_dtype':<12}  {'min':>12}  {'max':>12}  path")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*40}")
    for r in sorted(overflow_results, key=lambda x: x["min_val"]):
        print(
            f"  {r.get('header_dtype', '?'):<12}  "
            f"{r['min_val']:>12.1f}  "
            f"{r['max_val']:>12.1f}  "
            f"{r['rel_path']}"
        )

    # Save overflow file list as plain text (one relative path per line)
    os.makedirs(os.path.dirname(OVERFLOW_TXT), exist_ok=True)
    with open(OVERFLOW_TXT, "w") as f:
        for r in overflow_results:
            f.write(r["rel_path"] + "\n")
    print(f"\nOverflow file list saved → {OVERFLOW_TXT}")

    # ------------------------------------------------------------------
    # Step 5: Fix (only if --fix was requested)
    # ------------------------------------------------------------------
    if not args.fix:
        print("\n[INFO] Run with --fix to write dtype-corrected copies.")
        return

    # Only fix images where the hypothesis holds (header_dtype == int16)
    fixable = [r for r in overflow_results if r.get("header_dtype") == "int16"]
    not_fixable = [r for r in overflow_results if r.get("header_dtype") != "int16"]

    if not_fixable:
        print(f"\n[WARNING] {len(not_fixable)} overflow images have unexpected dtype "
              f"(not int16) and will be skipped:")
        for r in not_fixable:
            print(f"  dtype={r.get('header_dtype','?')}  {r['rel_path']}")

    print(
        f"\n[FIX] Writing uint16-corrected copies of {len(fixable)} images "
        f"to {args.output_dir} …"
    )
    fixed_ok: list[str] = []
    fixed_fail: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fix_entry, r, args.output_dir): r for r in fixable
        }
        for future in tqdm(as_completed(futures), total=len(fixable), unit="img"):
            success, msg = future.result()
            print(f"  {msg}")
            (fixed_ok if success else fixed_fail).append(msg)

    print(f"\n{'='*70}")
    print(f"  Fixed successfully: {len(fixed_ok):>6}")
    print(f"  Failed / skipped:   {len(fixed_fail):>6}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
