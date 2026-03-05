#!/usr/bin/env python3
"""
Script to download the OpenMind dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/AnonRes/OpenMind
  - 114k 3D Head-and-Neck MRI images in NIfTI format
  - 23 different MRI modalities from 800 OpenNeuro datasets
  - Associated deface_masks and anatomy (fb) masks
  - Metadata CSV: openneuro_metadata.csv

---------------------------------------------------------------------------
RECOMMENDED USAGE (single node, snapshot):
    python 0_download_openmind.py

MULTI-NODE USAGE (run this script on each SLURM node with a different rank):
    # Node 0 of 4:
    python 0_download_openmind.py --num-nodes 4 --node-rank 0
    # Node 1 of 4:
    python 0_download_openmind.py --num-nodes 4 --node-rank 1
    # ...

    Each node downloads its own shard of the 800 OpenNeuro sub-datasets.

SLURM EXAMPLE (see bash_download_openmind.sh):
    sbatch bash_download_openmind.sh

OTHER OPTIONS:
    --dry-run          List files without downloading
    --max-datasets N   Download only first N sub-datasets (for testing)
    --workers N        Parallel download threads per node (default: 32)
    --no-resume        Re-download even if file exists

Requirements:
    pip install huggingface_hub tqdm
---------------------------------------------------------------------------
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force unbuffered output (important for SLURM log files)
sys.stdout.reconfigure(line_buffering=True)

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_REPO_ID = "AnonRes/OpenMind"
REPO_TYPE = "dataset"

DEFAULT_OUTPUT_DIR = (
    "/lustre/fsw/portfolios/healthcareeng/projects/"
    "healthcareeng_monai/datasets/OpenMind"
)

PYTHON = (
    "/lustre/fsw/portfolios/healthcareeng/projects/"
    "healthcareeng_monai/MAISI/maisi_conda_env_v2/bin/python"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print with flush (essential for SLURM)."""
    print(msg, flush=True)


def list_repo_files(repo_id, token=None):
    """List all NIfTI and CSV files in the HuggingFace repo."""
    log("  Querying HuggingFace API for file list (may take 1-2 min for 114k files)...")
    api = HfApi()
    files = list(api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE, token=token))
    return sorted(files)


def get_dataset_ids(all_files):
    """
    Extract unique OpenNeuro sub-dataset IDs (ds_XXXXXX) from file paths.
    Returns sorted list.
    """
    ds_ids = set()
    for f in all_files:
        for part in Path(f).parts:
            if part.startswith("ds_"):
                ds_ids.add(part)
                break
    return sorted(ds_ids)


def shard_dataset_ids(all_ds_ids, num_nodes, node_rank):
    """
    Assign a shard of dataset IDs to this node.
    Node i gets indices: i, i+num_nodes, i+2*num_nodes, ...
    """
    return [ds for i, ds in enumerate(all_ds_ids) if i % num_nodes == node_rank]


def filter_files_for_datasets(all_files, allowed_ds_ids):
    """Keep only files belonging to the given ds_* IDs (plus top-level metadata)."""
    allowed = set(allowed_ds_ids)
    result = []
    for f in all_files:
        parts = Path(f).parts
        ds_part = next((p for p in parts if p.startswith("ds_")), None)
        if ds_part is None:
            result.append(f)   # top-level metadata / README
        elif ds_part in allowed:
            result.append(f)
    return result


def download_one(args_tuple):
    """Worker: download a single file. Returns (success, filename, message)."""
    repo_id, filename, output_dir, resume, max_retries, retry_delay, token = args_tuple
    local_path = os.path.join(output_dir, filename)

    if resume and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return True, filename, "skipped"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
            return True, filename, "downloaded"
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                return False, filename, f"FAILED: {exc}"

    return False, filename, "FAILED"


# ---------------------------------------------------------------------------
# Download modes
# ---------------------------------------------------------------------------

def run_snapshot(output_dir, resume, token):
    """Simplest approach: download the whole repo in one call."""
    log(f"Using snapshot_download() -> {output_dir}")
    local_dir = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=resume,
        token=token,
    )
    log(f"✓ Dataset downloaded to: {local_dir}")


def run_sharded(output_dir, num_nodes, node_rank, workers, resume,
                max_retries, retry_delay, max_datasets, dry_run, token):
    """Per-file parallel download, optionally sharded across nodes."""

    log("Listing repository files...")
    all_files = list_repo_files(DATASET_REPO_ID, token=token)
    log(f"  Total files in repo : {len(all_files)}")

    # Get all dataset IDs
    all_ds_ids = get_dataset_ids(all_files)
    log(f"  Total sub-datasets  : {len(all_ds_ids)}")

    # Apply max-datasets limit
    if max_datasets is not None:
        all_ds_ids = all_ds_ids[:max_datasets]
        log(f"  Limiting to first {max_datasets} sub-datasets")

    # Shard across nodes
    if num_nodes > 1:
        my_ds_ids = shard_dataset_ids(all_ds_ids, num_nodes, node_rank)
        log(f"  Node {node_rank}/{num_nodes}: assigned {len(my_ds_ids)} sub-datasets")
    else:
        my_ds_ids = all_ds_ids

    # Filter files for this node's shard
    files = filter_files_for_datasets(all_files, my_ds_ids)
    nii   = sum(1 for f in files if f.endswith(".nii.gz"))
    csv   = sum(1 for f in files if f.endswith(".csv"))
    log(f"  Files to download   : {len(files)}  ({nii} NIfTI, {csv} CSV)")

    if dry_run:
        log("\n[dry-run] First 20 files:")
        for f in files[:20]:
            log(f"  {f}")
        if len(files) > 20:
            log(f"  ... and {len(files) - 20} more")
        return

    os.makedirs(output_dir, exist_ok=True)

    task_args = [
        (DATASET_REPO_ID, f, output_dir, resume, max_retries, retry_delay, token)
        for f in files
    ]

    success_count = skipped_count = failed_count = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, a): a[1] for a in task_args}
        with tqdm(total=len(files), unit="file",
                  desc=f"Node {node_rank}", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                ok, fname, msg = future.result()
                if ok:
                    if msg == "skipped":
                        skipped_count += 1
                    else:
                        success_count += 1
                else:
                    failed_count += 1
                    failed_files.append((fname, msg))
                    tqdm.write(f"  ✗ {fname}: {msg}", file=sys.stderr)
                pbar.update(1)

    log("\n" + "=" * 60)
    log(f"Node {node_rank} Summary")
    log("=" * 60)
    log(f"  Downloaded : {success_count}")
    log(f"  Skipped    : {skipped_count}")
    log(f"  Failed     : {failed_count}")
    log("=" * 60)

    if failed_files:
        log("\nFailed files:")
        for fn, msg in failed_files:
            log(f"  {fn}: {msg}")

    if failed_count > 0:
        sys.exit(1)
    else:
        log(f"\n✓ Node {node_rank} done. Files saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the OpenMind 3D MRI dataset from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root directory to save the dataset.")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel download threads per node.")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Total number of SLURM nodes used for download.")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Rank of this node (0-indexed).")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip files that already exist (default: on).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-download all files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without downloading.")
    parser.add_argument("--max-datasets", type=int, default=None, metavar="N",
                        help="Limit to first N OpenNeuro sub-datasets.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retry attempts per file.")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                        help="Base delay (s) between retries.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Use snapshot_download() (simpler, no sharding).")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)

    log("=" * 60)
    log("OpenMind Dataset Downloader")
    log("=" * 60)
    log(f"  Repo       : {DATASET_REPO_ID}")
    log(f"  Output dir : {output_dir}")
    log(f"  Node       : {args.node_rank} / {args.num_nodes}")
    log(f"  Workers    : {args.workers}")
    log(f"  Resume     : {args.resume}")
    log(f"  Dry run    : {args.dry_run}")
    if args.snapshot:
        log("  Mode       : snapshot_download")
    else:
        log("  Mode       : sharded per-file")
    log("=" * 60)

    if args.snapshot:
        run_snapshot(output_dir, args.resume, args.hf_token)
    else:
        run_sharded(
            output_dir=output_dir,
            num_nodes=args.num_nodes,
            node_rank=args.node_rank,
            workers=args.workers,
            resume=args.resume,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_datasets=args.max_datasets,
            dry_run=args.dry_run,
            token=args.hf_token,
        )


if __name__ == "__main__":
    main()
