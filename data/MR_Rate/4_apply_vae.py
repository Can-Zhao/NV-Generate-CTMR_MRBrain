#!/usr/bin/env python3
"""
Script to apply trained VAE (autoencoder) to skull-stripped MR images.
MR-RATE dataset specific version with acquisition-plane-aware preprocessing.

WORKFLOW
========

For each image entry in the dataset JSON (which contains 'acquisition_plane'):

1. LOAD & ORIENT
   - Load the skull-stripped NIfTI image.
   - Reorient to RAS (Right–Anterior–Superior) standard orientation using MONAI.
   - Apply intensity normalization: scale 0–99.5th percentile to [0, 1].
   - Extract voxel spacing from the MetaTensor affine (correct after reorientation).

2. IDENTIFY ACQUISITION PLANE
   - Read 'acquisition_plane' from the JSON entry (axi / sag / cor / obl / None).
   - Map to in-plane and through-plane axis indices in RAS space:
       axi  →  in-plane axes: [0=R-L, 1=A-P],  through-plane axis: 2=S-I
       sag  →  in-plane axes: [1=A-P, 2=S-I],  through-plane axis: 0=R-L
       cor  →  in-plane axes: [0=R-L, 2=S-I],  through-plane axis: 1=A-P
   - For oblique ('obl') or unknown planes: fall back to the generic
     vae_utils.process_single_image logic, which selects multiple target sizes
     based on spacing similarity and encodes each resolution independently.
     Steps 3–5 below are skipped for oblique images.

3. MATCH IN-PLANE SPACING  (trilinear interpolation)
   - Let sp0 and sp1 be the spacing along the two in-plane axes.
   - fine_spacing = min(sp0, sp1)
   - For the coarser in-plane axis:
       new_dim = round(old_dim × old_spacing / fine_spacing)
   - Perform a 3-D trilinear interpolation to the updated target size;
     the through-plane dimension and spacing are left unchanged.
   - After this step, both in-plane axes share the same voxel spacing.

4. EQUALIZE IN-PLANE DIMENSIONS  (symmetric zero-padding)
   - Both in-plane axes now have equal spacing but may still differ in size.
   - max_ip_dim = max(dim_ax0, dim_ax1)
   - Zero-pad the smaller axis symmetrically (equal padding on each side,
     one extra voxel added to the right/bottom if the difference is odd).
   - After this step, both in-plane axes are equal in spacing AND dimension.

5. GENERATE TARGET SIZES + ENCODE  (vae_utils.encode_all_target_sizes)
   - For axi / sag / cor: the through-plane axis is already known from step 2,
     so vae_utils.identify_throughplane_axis is NOT called.
     vae_utils.encode_all_target_sizes is called with the preprocessed image,
     the effective spacing from steps 3–4, and the known through-plane axis.
     Internally it calls vae_utils.generate_target_3d_sizes_mri, which
     enumerates all valid target 3-D size tuples subject to one constraint:
       1. In-plane dimensions are equal.
     Then it calls vae_utils.encode_and_save for each candidate size.
   - For obl / unknown: vae_utils.process_single_image is called instead.
     It runs vae_utils.identify_throughplane_axis to determine the
     through-plane axis from spacing ratios, then also delegates to
     vae_utils.encode_all_target_sizes for the generate → encode loop.

6. SAVE  (inside vae_utils.encode_all_target_sizes → vae_utils.encode_and_save)
   - Each embedding is saved atomically as:
         <output_root>/<rel_dir>/<filename>_{X}x{Y}x{Z}_emb.nii.gz

7. REPORT
   - Each GPU prints a summary: success / skipped / error counts.
   - Errors are listed (first 10 shown) after processing.
"""

import os
import sys
import json
import argparse
import logging
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from monai.utils import set_determinism

# Add parent directory to path for vae_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from vae_utils import (
    download_autoencoder_model,
    load_autoencoder,
    create_preprocessing_transforms,
    load_and_get_spacing,
    encode_all_target_sizes,
    generate_target_3d_sizes_mri,
    is_valid_image_size_for_splits_80g_a100,
    process_single_image,
    cleanup_temp_files,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_JSON = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_brain_mask_pairs.json"
SKULL_STRIPPED_ROOT = (
    "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip_skull_stripped/mri/"
)
OUTPUT_EMBEDDING_ROOT = (
    "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/"
    "MAISI/data/encoding_128_downsample/MR-Rate/MR-RATE_20260227_unzip_skull_stripped/mri/"
)

# Axis mapping after RAS reorientation
# in_plane: list of two spatial axis indices that form the acquisition plane
# through_plane: axis index perpendicular to the acquisition plane
ACQUISITION_PLANE_AXES = {
    "axi": {"in_plane": [0, 1], "through_plane": 2},
    "sag": {"in_plane": [1, 2], "through_plane": 0},
    "cor": {"in_plane": [0, 2], "through_plane": 1},
}

# F.pad on a [C, X, Y, Z] tensor pads from the last spatial dim backward:
#   pad = (Z_left, Z_right, Y_left, Y_right, X_left, X_right)
_AXIS_TO_PAD_POSITIONS = {0: (4, 5), 1: (2, 3), 2: (0, 1)}

# Map spatial axis index → RAS axis name used by vae_utils functions
_AXIS_INDEX_TO_NAME = {0: 'x', 1: 'y', 2: 'z'}


# ── MR-RATE specific preprocessing ────────────────────────────────────────────


def preprocess_mrrate(image_tensor, spacing, acquisition_plane):
    """
    Apply MR-RATE acquisition-plane-aware preprocessing (steps 3–4 only).

    After this function the two in-plane axes have:
      • equal voxel spacing  = fine_spacing = min(sp_ax0, sp_ax1)
      • equal dimension size = max(new_dim_ax0, new_dim_ax1)

    The through-plane axis is left untouched.
    Subsequent target-size selection and VAE encoding are handled by the
    vae_utils functions (generate_target_3d_sizes_mri + encode_and_save).

    Args:
        image_tensor (torch.Tensor): Shape [C, X, Y, Z], float32.
        spacing (List[float]):       [sp_x, sp_y, sp_z] in RAS order (mm).
        acquisition_plane (str):     'axi', 'sag', or 'cor'.

    Returns:
        tuple:
            image_tensor (torch.Tensor): Preprocessed [C, X', Y', Z'], float32.
            effective_spacing (List[float]): Updated [sp_x, sp_y, sp_z] where
                both in-plane axes carry fine_spacing and the through-plane
                axis carries the original through-plane spacing.

    Raises:
        ValueError: If acquisition_plane is not supported.
    """
    axes_info = ACQUISITION_PLANE_AXES.get(acquisition_plane)
    if axes_info is None:
        raise ValueError(f"Unsupported acquisition_plane: '{acquisition_plane}'")

    ax0, ax1 = axes_info["in_plane"]   # two in-plane spatial axis indices
    ax_tp    = axes_info["through_plane"]
    sp = list(spacing)
    dims = list(image_tensor.shape[1:])  # [X, Y, Z]

    # ── Step 3: Match in-plane spacing ──────────────────────────────────────
    # Rarely triggered: most MRI acquisitions have isotropic in-plane spacing
    # (sp0 == sp1), making fine_sp == sp0 == sp1 and new_d == dims exactly.
    # Only fires for anisotropic in-plane protocols (e.g., rectangular pixels).
    sp0, sp1 = sp[ax0], sp[ax1]
    fine_sp  = min(sp0, sp1)

    new_d0 = round(dims[ax0] * sp0 / fine_sp)
    new_d1 = round(dims[ax1] * sp1 / fine_sp)

    target = list(dims)
    target[ax0] = new_d0
    target[ax1] = new_d1  # through-plane axis left untouched

    if target != dims:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),   # [1, C, X, Y, Z]
            size=target,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        dims = list(image_tensor.shape[1:])

    # ── Step 4: Equalize in-plane dimensions with zero-padding ──────────────
    # Rarely triggered: most scans have a square FOV so dims[ax0] == dims[ax1]
    # already. Only fires for rectangular FOVs or after step 3 produces unequal
    # dimensions from different original sizes.
    max_ip = max(dims[ax0], dims[ax1])

    pad_cfg = [0] * 6  # (Z_l, Z_r, Y_l, Y_r, X_l, X_r)
    for ax in (ax0, ax1):
        diff = max_ip - dims[ax]
        if diff > 0:
            pad_before = diff // 2
            pad_after  = diff - pad_before
            l_idx, r_idx = _AXIS_TO_PAD_POSITIONS[ax]
            pad_cfg[l_idx] = pad_before
            pad_cfg[r_idx] = pad_after

    if any(p > 0 for p in pad_cfg):
        image_tensor = F.pad(image_tensor, pad_cfg)

    # Build effective spacing: in-plane axes → fine_sp, through-plane unchanged.
    # list(spacing) already copies spacing[ax_tp], so no need to reassign it.
    effective_spacing = list(spacing)
    effective_spacing[ax0] = fine_sp
    effective_spacing[ax1] = fine_sp

    return image_tensor, effective_spacing


# ── Early-exit helpers ─────────────────────────────────────────────────────────


def _simulate_preprocess_dims(dims, spacing, acquisition_plane):
    """
    Mathematically simulate preprocess_mrrate steps 3–4 without loading any image.

    Uses the ``dim`` and ``spacing`` stored in the JSON (original NIfTI header
    values) to predict the effective dimensions and spacing after in-plane
    spacing equalisation (step 3) and in-plane dimension padding (step 4).

    Args:
        dims (list[int]):    Original image dimensions [x, y, z].
        spacing (list[float]): Original voxel spacing [sp_x, sp_y, sp_z] mm.
        acquisition_plane (str): 'axi', 'sag', or 'cor'.

    Returns:
        tuple: (eff_dims, eff_spacing) after simulation, or (None, None) if
            acquisition_plane is not a known plane.
    """
    axes_info = ACQUISITION_PLANE_AXES.get(acquisition_plane)
    if axes_info is None:
        return None, None

    ax0, ax1 = axes_info["in_plane"]
    ax_tp = axes_info["through_plane"]

    d  = list(dims)
    sp = list(spacing)

    # Step 3: match in-plane spacing to the finer of the two
    sp0, sp1 = sp[ax0], sp[ax1]
    fine_sp  = min(sp0, sp1)
    d[ax0]   = round(d[ax0] * sp0 / fine_sp)
    d[ax1]   = round(d[ax1] * sp1 / fine_sp)

    # Step 4: pad the smaller in-plane axis to equalise dimensions
    max_ip = max(d[ax0], d[ax1])
    d[ax0] = max_ip
    d[ax1] = max_ip

    # list(spacing) already copies spacing[ax_tp], so no need to reassign it.
    eff_sp = list(spacing)
    eff_sp[ax0] = fine_sp
    eff_sp[ax1] = fine_sp

    return d, eff_sp


def _all_outputs_exist(pair, output_root, num_splits):
    """
    Return True if every expected embedding for this image already exists on disk,
    using only the JSON metadata (``dim``, ``spacing``, ``acquisition_plane``).
    Sizes that exceed GPU memory are treated as "will be skipped anyway" and do
    not prevent an early exit.

    This avoids loading the NIfTI file when ``skip_existing=True`` and all work
    is already done.

    Uses a single ``os.scandir`` on the subject's output directory (one
    ``readdir`` RPC on Lustre) rather than one ``os.path.exists`` per target
    size (one ``stat`` RPC each).

    Args:
        pair (dict):       One JSON entry; must have 'image', 'dim', 'spacing',
                           and 'acquisition_plane'.
        output_root (str): Root directory for embedding outputs.
        num_splits (int):  Tensor-parallel split count (for the GPU size guard).

    Returns:
        bool: True only if all expected outputs are present (or GPU-oversized).
    """
    dims     = pair.get("dim")
    spacing  = pair.get("spacing")
    acq      = pair.get("acquisition_plane")
    img_rel  = pair.get("image", "")

    # Need dim/spacing in JSON and a known plane to pre-check
    if not dims or not spacing or acq not in ACQUISITION_PLANE_AXES:
        logger.debug("%s: _all_outputs_exist=False (missing dim/spacing or unknown plane: acq=%r)", img_rel, acq)
        return False

    eff_dims, eff_sp = _simulate_preprocess_dims(dims, spacing, acq)
    if eff_dims is None:
        return False

    through_plane_axis = _AXIS_INDEX_TO_NAME[ACQUISITION_PLANE_AXES[acq]["through_plane"]]
    try:
        target_sizes = generate_target_3d_sizes_mri(
            x_size=eff_dims[0],
            y_size=eff_dims[1],
            z_size=eff_dims[2],
            original_spacing=eff_sp,
            through_plane_axis=through_plane_axis,
        )
    except Exception:
        return False

    if not target_sizes:
        return False

    rel_dir  = os.path.dirname(img_rel)
    filename = os.path.basename(img_rel).replace(".nii.gz", "").replace(".nii", "")
    out_dir  = os.path.join(output_root, rel_dir)

    # Use per-file os.path.exists checks.  We avoid os.scandir caching because
    # it can hang indefinitely on Lustre under heavy concurrent I/O from
    # multiple GPUs.  Per-file checks are slower but guaranteed not to hang.
    for target_size in target_sizes:
        if not is_valid_image_size_for_splits_80g_a100(num_splits, target_size):
            continue
        x, y, z = target_size
        path = os.path.join(out_dir, f"{filename}_{x}x{y}x{z}_emb.nii.gz")
        if not os.path.exists(path):
            return False
    return True


# ── Per-image processing ───────────────────────────────────────────────────────

def process_mrrate_image(
    pair,
    skull_stripped_root,
    output_root,
    transforms,
    autoencoder,
    device,
    num_splits,
    skip_existing,
):
    """
    Full end-to-end pipeline for a single MR-RATE image entry.

    The ``pair`` dict is one entry from the dataset JSON produced by
    ``0_generate_img_brain_mask_pair_json.py``.  The two fields consumed here
    are:

    * ``"image"``             — relative path to the skull-stripped NIfTI.
    * ``"acquisition_plane"`` — ``'axi'``, ``'sag'``, ``'cor'``, ``'obl'``,
      or absent / ``None`` if the plane could not be parsed from the filename.

    **Known planes (axi / sag / cor)**

    1. Load, reorient to RAS, and intensity-normalise via ``transforms``;
       extract voxel spacing from the MetaTensor affine
       (``vae_utils.load_and_get_spacing``).
    2. Apply MR-RATE-specific preprocessing — match in-plane spacing and
       equalise in-plane dimensions via ``preprocess_mrrate``.
    3. Generate all valid target 3-D sizes and encode at each resolution via
       ``vae_utils.encode_all_target_sizes``.  The through-plane axis is taken
       directly from ``ACQUISITION_PLANE_AXES``; ``identify_throughplane_axis``
       is not called.

    **Oblique / unknown planes**

    Delegates entirely to ``vae_utils.process_single_image``, which runs
    ``vae_utils.identify_throughplane_axis`` on the spacing/dimensions and
    then calls ``vae_utils.encode_all_target_sizes``.

    Args:
        pair (dict): One JSON entry with at minimum ``"image"`` and optionally
            ``"acquisition_plane"``.
        skull_stripped_root (str): Absolute path to the root directory from
            which ``pair["image"]`` is a relative path.
        output_root (str): Root directory under which all embedding files are
            saved, mirroring the relative directory structure of the image.
        transforms (monai.transforms.Compose): Preprocessing pipeline from
            ``vae_utils.create_preprocessing_transforms()``.
        autoencoder (torch.nn.Module): Loaded autoencoder in eval mode.
        device (torch.device): CUDA device for encoding.
        num_splits (int): Tensor-parallel split count for the GPU memory guard.
        skip_existing (bool): If ``True``, skip target sizes whose output
            embedding already exists.

    Returns:
        tuple[int, int, int, list[str]]:
            - ``num_success``    — embeddings newly written.
            - ``num_errors``     — unrecoverable failures.
            - ``num_skipped``    — sizes skipped (existing file, OOM, etc.).
            - ``error_messages`` — human-readable error strings.
    """
    image_rel_path    = pair.get("image", "")
    acquisition_plane = pair.get("acquisition_plane")

    # ── Early-exit: check all outputs exist using JSON metadata only ──────────
    # Uses a single os.scandir on the subject's output directory (one readdir
    # RPC) rather than per-file os.path.exists (one stat RPC each), then does
    # O(1) set lookups for each expected filename.  Avoids opening the NIfTI
    # file entirely when all outputs are already on disk.
    # Falls back to full processing if any output is missing or if the JSON
    # lacks dim/spacing (e.g. oblique plane).
    if skip_existing and _all_outputs_exist(pair, output_root, num_splits):
        logger.debug("%s: all outputs exist, skipping image load.", image_rel_path)
        return 0, 0, 1, []

    image_full_path = os.path.join(skull_stripped_root, image_rel_path)

    if not os.path.exists(image_full_path):
        return 0, 1, 0, [f"Image not found: {image_full_path}"]

    # ── Oblique / unknown plane: fall back to vae_utils generic logic ────────
    if acquisition_plane not in ACQUISITION_PLANE_AXES:
        logger.info(
            "%s: acquisition_plane=%r, using generic vae_utils logic.",
            image_rel_path, acquisition_plane,
        )
        return process_single_image(
            image_full_path=image_full_path,
            image_rel_path=image_rel_path,
            output_root=output_root,
            transforms=transforms,
            autoencoder=autoencoder,
            device=device,
            num_splits=num_splits,
            skip_existing=skip_existing,
        )

    # ── Step 1: Load, orient, normalise and extract spacing ──────────────────
    try:
        image, spacing = load_and_get_spacing(image_full_path, image_rel_path, transforms)
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Preprocessing error: {e}"]

    orig_shape = list(image.shape[1:])
    ax_info    = ACQUISITION_PLANE_AXES[acquisition_plane]
    ax0, ax1   = ax_info["in_plane"]
    ax_tp      = ax_info["through_plane"]

    logger.debug(
        "%s: shape=%s, spacing=%s, plane=%s, in-plane spacing=(%.3f, %.3f)",
        image_rel_path, orig_shape, [round(s, 3) for s in spacing],
        acquisition_plane, spacing[ax0], spacing[ax1],
    )

    # ── Steps 3–4: MR-RATE resampling & padding ──────────────────────────────
    try:
        image, effective_spacing = preprocess_mrrate(image, spacing, acquisition_plane)
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: MR-RATE preprocessing error: {e}"]

    final_shape = list(image.shape[1:])  # [X, Y, Z] after all steps
    logger.debug(
        "%s: final shape=%s, effective spacing=%s",
        image_rel_path, final_shape, [round(s, 3) for s in effective_spacing],
    )

    # ── Steps 5–6: Generate target sizes and encode ──────────────────────────
    # Through-plane axis is already known from the acquisition plane metadata,
    # so identify_throughplane_axis is not needed here.
    through_plane_axis = _AXIS_INDEX_TO_NAME[ax_tp]
    rel_dir  = os.path.dirname(image_rel_path)
    filename = os.path.basename(image_rel_path).replace(".nii.gz", "").replace(".nii", "")
    out_filename_base = os.path.join(output_root, rel_dir, filename)

    return encode_all_target_sizes(
        image=image,
        effective_spacing=effective_spacing,
        through_plane_axis=through_plane_axis,
        image_rel_path=image_rel_path,
        out_filename_base=out_filename_base,
        autoencoder=autoencoder,
        device=device,
        num_splits=num_splits,
        skip_existing=skip_existing,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply VAE to skull-stripped MR images (MR-RATE, plane-aware)"
    )
    parser.add_argument(
        "--num-splits", "--num_splits",
        dest="num_splits", type=int, default=1,
        help="Number of splits for tensor parallelism (1, 2, 4, 8, or 16)",
    )
    parser.add_argument(
        "--no_skip_existing", "--no-skip-existing",
        dest="no_skip_existing", action="store_true",
        help="Reprocess all files (don't skip existing)",
    )
    args = parser.parse_args()

    # ── Distributed setup ────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()

    # Set NCCL timeout to 30 minutes (1800s) to allow slow GPUs time to finish encoding.
    # The default is 10 minutes (600s), which may be too short if some GPUs are still
    # encoding while others have finished (all skipped).
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    dist.init_process_group(backend="nccl", init_method="env://")
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    local_rank  = int(os.environ.get("LOCAL_RANK", global_rank))
    device      = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    set_determinism(seed=42)

    # Set logging level — change to logging.DEBUG to see per-image shape/spacing logs
    logging.basicConfig(level=logging.INFO)

    skip_existing = not args.no_skip_existing

    if global_rank == 0:
        logger.info("=" * 80)
        logger.info("MR-RATE VAE Application — acquisition-plane-aware preprocessing")
        logger.info("=" * 80)
        logger.info("Input JSON:            %s", INPUT_JSON)
        logger.info("Skull-stripped root:   %s", SKULL_STRIPPED_ROOT)
        logger.info("Output embedding root: %s", OUTPUT_EMBEDDING_ROOT)
        logger.info("Num splits:            %d", args.num_splits)
        logger.info("Skip existing:         %s", skip_existing)
        logger.info("World size:            %d", world_size)
        logger.info("=" * 80)
    # Log from one rank per node so you can confirm all nodes joined (e.g. 32 lines in log).
    print("Node joined: global_rank=%d (world_size=%d)", global_rank, world_size)

    # ── Clean up stale temp files from any previous interrupted run ──────────
    if global_rank == 0:
        cleanup_temp_files(OUTPUT_EMBEDDING_ROOT)
    dist.barrier()

    # ── Download & load autoencoder ──────────────────────────────────────────
    download_autoencoder_model(local_rank=global_rank)
    dist.barrier()

    if global_rank == 0:
        logger.info("Loading autoencoder...")
    autoencoder = load_autoencoder(device=device, num_splits=args.num_splits)

    # ── Preprocessing transforms ─────────────────────────────────────────────
    transforms = create_preprocessing_transforms()

    # ── Load JSON ────────────────────────────────────────────────────────────
    if global_rank == 0:
        logger.info("Loading input JSON: %s", INPUT_JSON)
    with open(INPUT_JSON, "r") as f:
        pairs = json.load(f)

    if global_rank == 0:
        logger.info("Total pairs in JSON: %d", len(pairs))

    # ── Randomize and distribute pairs across GPUs ───────────────────────────
    # Shuffle with a fixed seed so all ranks see the same order. Then assign via
    # round-robin (rank r gets indices r, r+world_size, r+2*world_size, ...) so
    # each rank gets a similar mix of easy/slow items and the final barrier sees
    # less load imbalance than with contiguous chunks.
    # random.seed(42)
    random.shuffle(pairs)
    my_pairs = [pairs[i] for i in range(global_rank, len(pairs), world_size)]

    if global_rank == 0:
        logger.info("Randomized pair order (seed=42), round-robin assignment across %d ranks", world_size)
        logger.info("GPU 0: processing %d pairs (indices 0, %d, %d, ...)", len(my_pairs), world_size, 2 * world_size)
        logger.info("=" * 80)

    # ── Process ──────────────────────────────────────────────────────────────
    total_success = total_errors = total_skipped = 0
    all_error_messages = []

    for pair in tqdm(my_pairs, desc=f"GPU {global_rank}", disable=(global_rank != 0)):
        n_ok, n_err, n_skip, msgs = process_mrrate_image(
            pair=pair,
            skull_stripped_root=SKULL_STRIPPED_ROOT,
            output_root=OUTPUT_EMBEDDING_ROOT,
            transforms=transforms,
            autoencoder=autoencoder,
            device=device,
            num_splits=args.num_splits,
            skip_existing=skip_existing,
        )
        total_success  += n_ok
        total_errors   += n_err
        total_skipped  += n_skip
        all_error_messages.extend(msgs)

    # ── Summary ──────────────────────────────────────────────────────────────
    if global_rank == 0:
        logger.info("=" * 80)
        logger.info("GPU %d Summary:", global_rank)
        logger.info("=" * 80)
        logger.info("  Successfully encoded: %d", total_success)
        logger.info("  Skipped:              %d", total_skipped)
        logger.info("  Errors:               %d", total_errors)

        if all_error_messages:
            logger.info("Error messages (first 10):")
            for msg in all_error_messages[:10]:
                logger.error("  - %s", msg)
            if len(all_error_messages) > 10:
                logger.info("  ... and %d more errors", len(all_error_messages) - 10)

        logger.info("=" * 80)
        logger.info("Done!")
        logger.info("=" * 80)

    # ── Synchronize all GPUs before exit ────────────────────────────────────────
    # Fast GPUs (all skipped) wait here for slow GPUs (still encoding) to finish.
    # NOTE: If any GPU is truly hung (not just slow), this barrier will timeout
    # after PyTorch's NCCL timeout (default 600s). The timeout indicates a hung
    # GPU, not a slow one.
    logger.info("GPU %d: Waiting for all GPUs to finish...", global_rank)
    try:
        dist.barrier()
        logger.info("GPU %d: All GPUs finished. Exiting...", global_rank)
    except Exception as e:
        logger.error("GPU %d: Barrier failed: %s", global_rank, str(e))
        raise

    # process group cleaned up on exit


if __name__ == "__main__":
    main()
