#!/usr/bin/env python3
"""
Shared utilities for VAE encoding of 3-D MRI volumes.

This module centralises logic that is reused across dataset-specific scripts
(e.g. ``MR_Rate/3_apply_vae.py``) so that changes only need to be made in one
place.

Public API
----------
Model helpers
~~~~~~~~~~~~~
download_autoencoder_model(local_rank)
    Download the autoencoder checkpoint on rank-0 if it is not already present.
load_autoencoder(device, num_splits)
    Instantiate and load the autoencoder from the project config + checkpoint.
create_preprocessing_transforms()
    Build the MONAI ``Compose`` pipeline for MRI preprocessing (orient → RAS,
    percentile intensity normalisation).

Size / spacing utilities
~~~~~~~~~~~~~~~~~~~~~~~~
get_spacing_from_metatensor(image)
    Extract RAS-aligned voxel spacing from a MONAI MetaTensor affine (correct
    after ``Orientationd``; preferred over reading the original NIfTI header).
load_and_get_spacing(image_full_path, image_rel_path, transforms)
    Load a NIfTI file through the preprocessing pipeline and return
    ``(image, spacing)``.  Shared by all image-processing entry points.
get_downsample_sizes(current_size, possible_sizes)
    Return all candidate downsample sizes that are ≤ ``current_size``.
identify_throughplane_axis(x_size, y_size, z_size, original_spacing)
    Determine which RAS axis is the through-plane (slice-encoding) direction
    from voxel spacings; falls back to dimension-size similarity.
generate_target_3d_sizes_mri(x_size, y_size, z_size, original_spacing,
                              through_plane_axis, possible_sizes)
    Enumerate valid 3-D target size tuples given an explicit through-plane axis.
    Enforces one constraint:
      1. In-plane dimensions are equal (e.g. ``[256, 256, 128]``).
check_spacing_constraint(original_shape, original_spacing, target_size)
    Post-hoc check that a given target size satisfies the spacing constraint.
    *Note*: this constraint is no longer enforced by ``generate_target_3d_sizes_mri``.

Encoding pipeline
~~~~~~~~~~~~~~~~~
is_valid_image_size_for_splits_80g_a100(num_splits, image_size)
    GPU-memory guard: return whether a given 3-D size fits in an 80 GB A100
    with the requested number of tensor-parallel splits.
encode_and_save(image, out_filename_base, target_size, autoencoder, device,
                num_splits, skip_existing)
    Downsample an image to ``target_size``, encode it with the VAE, and save
    the latent embedding as a NIfTI file.
encode_all_target_sizes(image, effective_spacing, through_plane_axis,
                        image_rel_path, out_filename_base, autoencoder, device,
                        num_splits, skip_existing)
    Generate all valid target 3-D sizes and call ``encode_and_save`` for each.
    Shared by ``process_single_image`` and dataset-specific callers (e.g.
    ``MR_Rate/3_apply_vae.py``) to avoid duplicating the generate → encode loop.
process_single_image(image_full_path, image_rel_path, output_root, transforms,
                     autoencoder, device, num_splits, skip_existing)
    End-to-end pipeline for one MRI file: preprocess → identify through-plane
    axis → encode all target sizes via ``encode_all_target_sizes``.
"""

import os
import sys
import json
import logging
import torch
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Resize,
    EnsureTyped,
)

# Add scripts to path for download_model_data and utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../scripts"))
from download_model_data import download_model_data
from utils import define_instance

# Configuration
# Use absolute paths to avoid issues with working directory in multi-node setups
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of 'data' directory
AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "models/autoencoder_v1.pt")
NETWORK_CONFIG = os.path.join(PROJECT_ROOT, "configs/config_network_rflow.json")
POSSIBLE_SIZES = [128, 256, 384, 512, 640, 768]
MIN_SIZE = 32


def download_autoencoder_model(local_rank=0):
    """
    Download the autoencoder checkpoint if it is not already present.

    Only rank 0 performs the download; other ranks should call this function
    after a barrier so they see the completed file.

    Args:
        local_rank (int): Local process rank. Download is skipped for ranks != 0.

    Returns:
        str: Absolute path to the autoencoder checkpoint file.

    Raises:
        FileNotFoundError: If the download completes but the file is still absent.
    """
    if local_rank == 0:
        if os.path.exists(AUTOENCODER_PATH):
            logger.info("Autoencoder model already exists at %s", AUTOENCODER_PATH)
            return AUTOENCODER_PATH
        
        logger.info("Downloading autoencoder_v1.pt model...")
        # Download using rflow-ct which includes autoencoder_v1.pt
        download_model_data(generate_version="rflow-ct", root_dir=PROJECT_ROOT, model_only=True)
        
        if not os.path.exists(AUTOENCODER_PATH):
            raise FileNotFoundError(f"Failed to download autoencoder model to {AUTOENCODER_PATH}")
        
        logger.info("Autoencoder v1 model downloaded to %s", AUTOENCODER_PATH)
    
    return AUTOENCODER_PATH


def load_autoencoder(device, num_splits=1):
    """
    Instantiate and load the trained autoencoder from the project config and checkpoint.

    Follows the same pattern as ``inference_tutorial.ipynb``: the network
    architecture is constructed via ``define_instance`` from the JSON config,
    then weights are loaded from ``AUTOENCODER_PATH``.

    Args:
        device (torch.device): Target device for the model.
        num_splits (int): Number of tensor-parallel splits to use inside the
            autoencoder.  Overrides the value in the config file.  Larger
            values reduce per-device memory usage at the cost of communication
            overhead.

    Returns:
        torch.nn.Module: Autoencoder model set to ``eval()`` mode, on ``device``.
    """
    logger.info("Loading autoencoder from %s", AUTOENCODER_PATH)
    logger.info("Using network config: %s", NETWORK_CONFIG)
    
    # Load network configuration
    with open(NETWORK_CONFIG, 'r') as f:
        network_config = json.load(f)
    
    # Override num_splits if specified
    if num_splits is not None:
        network_config["autoencoder_def"]["num_splits"] = num_splits
        logger.info("Overriding num_splits to %d", num_splits)
    
    # Create args namespace for define_instance (similar to inference_tutorial.ipynb)
    class Args:
        pass
    
    args = Args()
    # Copy all config values to args
    for k, v in network_config.items():
        setattr(args, k, v)
    
    # Define autoencoder using config (same as inference_tutorial.ipynb line 163)
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    
    # Load checkpoint
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device)
    if "unet_state_dict" in checkpoint:
        autoencoder.load_state_dict(checkpoint["unet_state_dict"])
    else:
        autoencoder.load_state_dict(checkpoint)
    
    autoencoder.eval()
    
    logger.info("Autoencoder loaded successfully (%s)", autoencoder.__class__.__name__)
    
    return autoencoder


def create_preprocessing_transforms():
    """
    Build the MONAI preprocessing pipeline for MRI volumes.

    The pipeline (based on ``diffusion_model_scale_up/data/1_resample_128_v1_1.py``)
    applies the following steps in order:

    1. **Load** — read the NIfTI file and ensure the channel dimension is first.
    2. **Orient** — reorient to RAS+ so spatial axes are consistent across
       subjects (x = left→right, y = posterior→anterior, z = inferior→superior).
    3. **Intensity normalise** — linearly map the [0th, 99.5th] percentile
       intensity range to [0, 1] (no clipping).
    4. **Type cast** — convert to ``float32`` while preserving MONAI metadata.

    Returns:
        monai.transforms.Compose: Composed transform pipeline.  Apply it to a
        dict with key ``"image"`` containing the path to the NIfTI file.
    """
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Orientationd(keys=["image"], axcodes="RAS"),
        # MRI intensity normalization: scale 0-99.5 percentile to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=["image"], 
            lower=0.0, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=False
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=True),
    ])
    
    return transforms


def get_spacing_from_metatensor(image):
    """
    Extract voxel spacing (mm) from a MONAI MetaTensor affine.

    The column norms of the first 3×3 sub-matrix of the affine give the voxel
    spacing along each spatial axis.  Because the affine is already in RAS order
    after ``Orientationd``, the result is correctly aligned with the (possibly
    reoriented) axes — unlike ``nib.header.get_zooms()``, which reflects the
    original on-disk axis order and can therefore be misaligned after
    reorientation.

    Args:
        image (MetaTensor): Image tensor with shape ``[C, X, Y, Z]`` and a
            valid ``.affine`` attribute, as returned by
            ``create_preprocessing_transforms()``.

    Returns:
        list[float]: ``[sp_x, sp_y, sp_z]`` — voxel spacing in mm along each
        RAS axis after reorientation.
    """
    affine = image.affine
    if hasattr(affine, "numpy"):
        affine = affine.numpy()
    else:
        affine = np.array(affine)
    return [float(np.linalg.norm(affine[:3, i])) for i in range(3)]


def is_valid_image_size_for_splits_80g_a100(num_splits, image_size):
    """
    Return whether a 3-D image fits in an 80 GB A100 with the given tensor-parallel splits.

    Volume thresholds are empirically derived from
    ``diffusion_model_scale_up/data/1_resample_128_v1_1.py`` and assume
    ``float16`` activations during the autoencoder forward pass.

    Args:
        num_splits (int): Number of tensor-parallel splits — must be one of
            ``{1, 2, 4, 8, 16}``.
        image_size (sequence of int): Spatial dimensions ``[x, y, z]``.

    Returns:
        bool: ``True`` if the volume fits within the GPU memory budget.

    Raises:
        ValueError: If ``num_splits`` is not a supported value.
    """
    volume = image_size[0] * image_size[1] * image_size[2]
    
    if num_splits == 1:
        return volume < 512 * 512 * 128
    elif num_splits == 2:
        return volume < 768 * 768 * 128
    elif num_splits == 4:
        return volume < 512 * 512 * 512
    elif num_splits == 8:
        return volume < 512 * 512 * 768
    elif num_splits == 16:
        return True  # volume >= 512 * 512 * 768
    else:
        raise ValueError("Invalid num_splits value")


def get_downsample_sizes(current_size, possible_sizes=POSSIBLE_SIZES):
    """
    Return all candidate sizes in ``possible_sizes`` that are ≤ ``current_size``.

    ``current_size`` is first rounded up to the nearest multiple of 128 that
    appears in ``possible_sizes``; if that value still exceeds the list, it is
    stepped down by 128 until a valid entry is found.

    Based on ``get_downsample_sizes()`` from
    ``diffusion_model_scale_up/data/1_resample_128_v1_1.py``.

    Args:
        current_size (int): Original size of one spatial dimension in voxels.
        possible_sizes (list[int]): Ordered list of candidate sizes, e.g.
            ``[128, 256, 384, 512]``.

    Returns:
        list[int]: Subset of ``possible_sizes`` whose values are ≤ the
        (rounded) ``current_size``, sorted in ascending order.

    Raises:
        ValueError: If ``current_size`` is smaller than ``MIN_SIZE``, or if
            rounding down exhausts all candidates below the minimum.
    """
    import math
    
    # Ensure the sizes are sorted in ascending order
    possible_sizes = sorted(possible_sizes)
    
    current_size = int(current_size)
    
    # Check if current_size is too small
    if current_size < MIN_SIZE:
        raise ValueError(f"Image dimension {current_size} too small (min: {MIN_SIZE}).")
    
    if current_size < min(possible_sizes):
        current_size = min(possible_sizes)
    
    # Round up to nearest multiple of 128 if not in possible_sizes
    if current_size not in possible_sizes:
        current_size = int(math.ceil(current_size / 128.) * 128)
    
    # If still not in possible_sizes, round down
    while current_size not in possible_sizes:
        current_size -= 128
        if current_size < min(possible_sizes):
            raise ValueError(f"Current size after rounding is smaller than minimum.")
    
    # Return all sizes up to and including current_size
    current_index = possible_sizes.index(current_size)
    return possible_sizes[:current_index + 1]


def identify_throughplane_axis(x_size, y_size, z_size, original_spacing, image_rel_path=""):
    """
    Identify the through-plane (slice-encoding) axis from voxel spacing.

    Strategy (in priority order):

    1. **Two spacings exactly equal, one different** — the unique (singleton)
       axis is the through-plane axis.  This is the most reliable indicator:
       standard 2-D MRI has identical in-plane pixel sizes and a distinct
       (usually coarser) slice thickness.

    2. **All three spacings different** — select the axis with the coarsest
       (largest) spacing.  The through-plane direction in 2-D MRI is always
       sampled more coarsely than the in-plane directions; for near-isotropic
       3-D acquisitions a warning is emitted.

    Args:
        x_size, y_size, z_size (int): Image dimensions along RAS axes
            (unused here but kept for API compatibility).
        original_spacing (List[float]): Voxel spacing [sp_x, sp_y, sp_z]
            in mm.
        image_rel_path (str): Relative path of the image, used as a prefix
            in warning messages for easier identification. Defaults to ``""``.

    Returns:
        str: Name of the through-plane axis — one of ``'x'``, ``'y'``,
        ``'z'``.
    """
    sp_x, sp_y, sp_z = original_spacing[0], original_spacing[1], original_spacing[2]
    spacings = [sp_x, sp_y, sp_z]
    axis_names = ['x', 'y', 'z']
    prefix = f"{image_rel_path}: " if image_rel_path else ""

    # Tolerance for treating two spacings as "equal" (relative ratio threshold).
    # If max(a, b) / min(a, b) < EQUAL_RATIO_THRESH, the two are considered equal.
    EQUAL_RATIO_THRESH = 1.0000001
    # Threshold for warning about near-isotropic spacings in Case 2.
    # If max / second_max < ISOTROPIC_WARN_THRESH, the coarsest-axis choice is uncertain.
    ISOTROPIC_WARN_THRESH = 1.1

    def _nearly_equal(a: float, b: float) -> bool:
        """Return True if a and b are within the relative tolerance."""
        if a <= 0 or b <= 0:
            return a == b
        lo, hi = min(a, b), max(a, b)
        return hi / lo < EQUAL_RATIO_THRESH

    xy_eq = _nearly_equal(sp_x, sp_y)
    xz_eq = _nearly_equal(sp_x, sp_z)
    yz_eq = _nearly_equal(sp_y, sp_z)

    # ── Case 1: two spacings nearly equal, one clearly different ──────────────
    # The singleton (unique) axis is the through-plane direction.
    # e.g. [0.9, 0.9001, 5.0] → z is through-plane  (sp_x ≈ sp_y, sp_z unique)
    if xy_eq and not xz_eq and not yz_eq:
        tp_idx = 2  # z is singleton
        logger.debug(
            "%sidentify_throughplane_axis: sp_x≈sp_y (%.4f≈%.4f), sp_z=%.4f → through-plane='z'",
            prefix, sp_x, sp_y, sp_z,
        )
    elif xz_eq and not xy_eq and not yz_eq:
        tp_idx = 1  # y is singleton
        logger.debug(
            "%sidentify_throughplane_axis: sp_x≈sp_z (%.4f≈%.4f), sp_y=%.4f → through-plane='y'",
            prefix, sp_x, sp_z, sp_y,
        )
    elif yz_eq and not xy_eq and not xz_eq:
        tp_idx = 0  # x is singleton
        logger.debug(
            "%sidentify_throughplane_axis: sp_y≈sp_z (%.4f≈%.4f), sp_x=%.4f → through-plane='x'",
            prefix, sp_y, sp_z, sp_x,
        )
    else:
        # ── Case 2: all three differ (or all three nearly equal) → coarsest ──
        tp_idx = int(np.argmax(spacings))
        sorted_sp = sorted(spacings, reverse=True)
        if sorted_sp[1] > 0 and sorted_sp[0] / sorted_sp[1] < ISOTROPIC_WARN_THRESH:
            logger.warning(
                "%sidentify_throughplane_axis: spacings %s are near-isotropic "
                "(ratio=%.3f < %.3f). Selecting coarsest axis '%s' as through-plane.",
                prefix, spacings, sorted_sp[0] / sorted_sp[1], ISOTROPIC_WARN_THRESH, axis_names[tp_idx],
            )
        else:
            logger.debug(
                "%sidentify_throughplane_axis: spacings %s all differ → "
                "coarsest axis '%s' as through-plane.",
                prefix, spacings, axis_names[tp_idx],
            )

    return axis_names[tp_idx]


def generate_target_3d_sizes_mri(x_size, y_size, z_size, original_spacing,
                                  through_plane_axis, possible_sizes=POSSIBLE_SIZES):
    """
    Generate all possible 3D target sizes for MRI with constraint:
      1. In-plane dims are equal: valid patterns [x, x, z], [x, z, x], [z, x, x].

    Enumerates all valid downsample combinations for the in-plane (paired)
    and through-plane (single) dimensions given an explicit through-plane axis.
    Use ``identify_throughplane_axis`` to determine the axis automatically.

    Args:
        x_size, y_size, z_size (int): Original image dimensions along RAS axes.
        original_spacing (List[float]): Voxel spacing [sp_x, sp_y, sp_z] in mm.
        through_plane_axis (str): The axis that is the through-plane direction —
            one of ``'x'``, ``'y'``, ``'z'``.
        possible_sizes: List of candidate sizes.

    Returns:
        List of target 3D size tuples [(x1,y1,z1), (x2,y2,z2), ...]
    """
    # Map axis name → (original_size, original_spacing) for convenient lookup
    axis_info = {
        'x': (x_size, original_spacing[0]),
        'y': (y_size, original_spacing[1]),
        'z': (z_size, original_spacing[2]),
    }

    in_plane_dims = tuple(ax for ax in ('x', 'y', 'z') if ax != through_plane_axis)

    through_plane_size = axis_info[through_plane_axis][0]
    in_plane_size = min(axis_info[ax][0] for ax in in_plane_dims)

    # Get downsample sizes for in-plane and through-plane dimensions
    in_plane_targets = get_downsample_sizes(in_plane_size, possible_sizes)
    through_plane_targets = get_downsample_sizes(through_plane_size, possible_sizes)

    # Ensure at least one value per dimension
    if not in_plane_targets:
        in_plane_targets = [min(possible_sizes)]
    if not through_plane_targets:
        through_plane_targets = [min(possible_sizes)]

    # Generate target sizes satisfying the constraint
    target_3d_sizes_set = set()

    for in_plane_val in in_plane_targets:
        for through_plane_val in through_plane_targets:
            size_dict = {ax: in_plane_val for ax in in_plane_dims}
            size_dict[through_plane_axis] = through_plane_val
            target_3d_sizes_set.add((size_dict['x'], size_dict['y'], size_dict['z']))

    return sorted(target_3d_sizes_set)


def check_spacing_constraint(original_shape, original_spacing, target_size):
    """
    Check whether a target size satisfies the through-plane spacing constraint.

    The constraint requires that, after resampling, the through-plane (singleton)
    spacing is ≥ the in-plane (paired) spacing::

        resampled_through_plane_spacing >= resampled_in_plane_spacing

    The in-plane and through-plane axes are inferred structurally: the two
    target dimensions that share the same value are treated as the in-plane
    axes, and the remaining dimension is treated as the through-plane axis.

    .. note::
        This function is provided as a standalone utility for callers that
        generate target sizes through other means and want to check this constraint.
        Note that :func:`generate_target_3d_sizes_mri` no longer enforces this constraint.

    Args:
        original_shape (sequence[int]): Original voxel dimensions
            ``[dim_x, dim_y, dim_z]``.
        original_spacing (sequence[float]): Original voxel spacing in mm
            ``[sp_x, sp_y, sp_z]``.
        target_size (sequence[int]): Target voxel dimensions
            ``[dim_x, dim_y, dim_z]`` after resampling.

    Returns:
        bool: ``True`` if the constraint is satisfied or cannot be determined
        (e.g. all three target dimensions are equal or all different).
    """
    from collections import Counter
    
    # Compute new spacing after resampling
    new_spacing = [
        original_spacing[i] * (original_shape[i] / target_size[i])
        for i in range(3)
    ]
    
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


def encode_and_save(image, out_filename_base, target_size, autoencoder, device, num_splits,
                    skip_existing=True):
    """
    Downsample an image to ``target_size``, encode it with the VAE, and save the latent.

    Based on ``downsample_and_save()`` from
    ``diffusion_model_scale_up/data/1_resample_128_v1_1.py``.

    The latent tensor produced by ``autoencoder.encode_stage_2_inputs`` has
    shape ``[1, 4, x/4, y/4, z/4]``.  It is squeezed, transposed to
    ``[x/4, y/4, z/4, 4]``, cast to ``float32``, and saved as a NIfTI file at::

        {out_filename_base}_{x}x{y}x{z}_emb.nii.gz

    The write is atomic: the file is first saved to a ``.tmp.nii.gz`` path and
    then renamed to prevent downstream readers from seeing a partial file.

    Args:
        image (MetaTensor): Preprocessed image tensor with shape ``[1, H, W, D]``
            and MONAI metadata (affine, etc.), as returned by
            ``create_preprocessing_transforms()``.
        out_filename_base (str): Base output path, without size suffix or
            extension (e.g. ``/output/subject_001``).
        target_size (tuple[int, int, int]): Target spatial dimensions
            ``(x, y, z)`` to resample to before encoding.
        autoencoder (torch.nn.Module): Loaded autoencoder model in eval mode.
        device (torch.device): Device on which to run the encoder.
        num_splits (int): Tensor-parallel split count passed to the GPU memory
            guard ``is_valid_image_size_for_splits_80g_a100``.
        skip_existing (bool): If ``True``, skip encoding when the output file
            already exists.  The atomic write pattern guarantees completeness,
            so no load-to-verify step is needed.

    Returns:
        tuple[bool, str, str | None]:
            - ``success`` — ``True`` if the embedding was saved (or skipped as
              already existing).
            - ``message`` — Human-readable status string.
            - ``output_path`` — Absolute path to the saved NIfTI file, or
              ``None`` on failure.
    """
    x_target, y_target, z_target = target_size

    # Define save path
    save_path = f"{out_filename_base}_{x_target}x{y_target}x{z_target}_emb.nii.gz"

    # Check if output already exists. The atomic write (temp file + os.replace)
    # guarantees that if the file exists it was written completely, so no
    # load-to-verify step is needed.
    if skip_existing and os.path.exists(save_path):
        return True, "Already exists", save_path
    
    # Check if size is valid for GPU memory
    if not is_valid_image_size_for_splits_80g_a100(num_splits=num_splits, image_size=target_size):
        return False, f"Image size too large for num_splits={num_splits}", None
    
    try:
        # Get original shape
        original_shape = image.shape[1:]  # [H, W, D]
        
        # Resample to target size
        resampler = Resize(spatial_size=target_size, mode='trilinear')
        downsampled_image = resampler(image)
        
        # Get affine for saving
        new_affine = downsampled_image.affine if hasattr(downsampled_image, 'affine') else np.eye(4)
        
        # Prepare for VAE encoding: convert to half precision and add batch dim
        downsampled_image_tensor = downsampled_image.half().unsqueeze(0).to(device)
        
        # Encode with VAE
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                try:
                    z = autoencoder.encode_stage_2_inputs(downsampled_image_tensor)
                except RuntimeError as e:
                    # Try processing in smaller batches if OOM
                    if "out of memory" in str(e).lower():
                        return False, f"OOM error: {str(e)}", None
                    else:
                        raise e
        
        # Save embedding using atomic write (write to temp file, then rename)
        # z shape: [1, 4, x/4, y/4, z/4] -> transpose to [x/4, y/4, z/4, 4] for NIfTI
        z_np = z.squeeze(0).cpu().detach().numpy().transpose(1, 2, 3, 0)
        z_nifti = nib.Nifti1Image(np.float32(z_np), new_affine)
        
        # Create output directory (with retry for distributed environments)
        output_dir = os.path.dirname(save_path)
        if output_dir:  # Only create if directory path is non-empty
            try:
                os.makedirs(output_dir, exist_ok=True)
            except FileExistsError:
                # Directory was created by another process, that's fine
                pass
        
        # Atomic write: save to temp file first, then rename
        # Use a proper .nii.gz extension for the temp file so nibabel can recognize it
        temp_save_path = save_path.replace('.nii.gz', '.tmp.nii.gz')
        try:
            nib.save(z_nifti, temp_save_path)
            # Atomic rename - if this succeeds, file is complete
            os.replace(temp_save_path, save_path)
        except Exception as e:
            # Clean up temp file if save failed
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except:
                    pass
            raise e
        
        return True, f"Saved to {save_path}", save_path
        
    except Exception as e:
        return False, f"Error: {str(e)}", None


def cleanup_temp_files(output_root):
    """
    Remove stale ``.tmp.nii.gz`` files left by interrupted encoding runs.

    ``encode_and_save`` writes to a temp file first and then renames it
    atomically.  If the process is killed between these two steps the temp
    file is left on disk.  It is safe to delete these because:

    * A temp file that exists means the final ``_emb.nii.gz`` was **not**
      written (the rename never happened), so nothing valid is lost.
    * A temp file is never read by any downstream code.

    This function should be called once at startup from rank 0, before
    ``dist.barrier()``, so all ranks see a clean output directory.

    Args:
        output_root (str): Root directory to search recursively for
            ``*.tmp.nii.gz`` files.

    Returns:
        int: Number of temp files removed.
    """
    removed = 0
    for dirpath, _, filenames in os.walk(output_root):
        for fname in filenames:
            if fname.endswith(".tmp.nii.gz"):
                fpath = os.path.join(dirpath, fname)
                try:
                    os.remove(fpath)
                    logger.info("Removed stale temp file: %s", fpath)
                    removed += 1
                except Exception as e:
                    logger.warning("Could not remove temp file %s: %s", fpath, e)
    if removed:
        logger.info("Cleaned up %d stale temp file(s) under %s", removed, output_root)
    else:
        logger.debug("No stale temp files found under %s", output_root)
    return removed


def load_and_get_spacing(image_full_path, image_rel_path, transforms):
    """
    Load a NIfTI image through the preprocessing pipeline and extract voxel spacing.

    Encapsulates the steps common to every image-processing entry point:

    1. Apply ``transforms`` (load → orient to RAS → intensity normalise → cast
       to float32), producing a MONAI MetaTensor.
    2. Extract voxel spacing from the RAS-reoriented MetaTensor affine via
       ``get_spacing_from_metatensor``.  Falls back to ``[1.0, 1.0, 1.0]``
       with a warning if the affine is unavailable.

    Args:
        image_full_path (str): Absolute path to the NIfTI file.
        image_rel_path (str): Relative path used in log and error messages.
        transforms (monai.transforms.Compose): Preprocessing pipeline from
            ``create_preprocessing_transforms()``.

    Returns:
        tuple[MetaTensor, list[float]]:
            - ``image``   — preprocessed tensor, shape ``[C, X, Y, Z]``.
            - ``spacing`` — voxel spacing ``[sp_x, sp_y, sp_z]`` in mm.

    Raises:
        Exception: Re-raises any exception thrown by ``transforms``
            (e.g. file not found, corrupt NIfTI).
    """
    preprocessed = transforms({"image": image_full_path})
    image = preprocessed["image"]

    try:
        spacing = get_spacing_from_metatensor(image)
    except Exception:
        logger.warning("%s: Could not extract spacing, using [1.0, 1.0, 1.0]", image_rel_path)
        spacing = [1.0, 1.0, 1.0]

    return image, spacing


def encode_all_target_sizes(image, effective_spacing, through_plane_axis,
                             image_rel_path, out_filename_base,
                             autoencoder, device, num_splits, skip_existing=True):
    """
    Generate all valid target 3-D sizes and encode the image at each one.

    Combines :func:`generate_target_3d_sizes_mri` and :func:`encode_and_save`
    into a single convenience wrapper.  Both ``process_single_image`` and
    dataset-specific callers (e.g. ``MR_Rate/4_apply_vae.py``) should call
    this function rather than re-implementing the generate → encode loop.

    The main skip-optimisation lives one level up: ``_all_outputs_exist`` in
    ``4_apply_vae.py`` uses a single ``os.scandir`` per subject directory to
    check whether *all* expected outputs already exist, avoiding the NIfTI
    load entirely for already-processed images.  This function's per-size
    ``os.path.exists`` check (inside ``encode_and_save``) is only reached for
    images with partial or no completion.

    Args:
        image (MetaTensor): Preprocessed image tensor with shape ``[C, X, Y, Z]``
            and MONAI metadata, as returned by
            ``create_preprocessing_transforms()`` (optionally after further
            dataset-specific preprocessing such as in-plane equalization).
        effective_spacing (list[float]): Voxel spacing ``[sp_x, sp_y, sp_z]``
            in mm, reflecting any resampling already applied to ``image``.
        through_plane_axis (str): Through-plane axis name — ``'x'``, ``'y'``,
            or ``'z'``.
        image_rel_path (str): Relative path used for error message prefixing
            and log output.
        out_filename_base (str): Base output path without size suffix or
            extension (e.g. ``/output/subject_001``).
        autoencoder (torch.nn.Module): Loaded autoencoder in eval mode.
        device (torch.device): Encoding device.
        num_splits (int): Tensor-parallel split count for the GPU memory guard.
        skip_existing (bool): If ``True``, skip target sizes whose embedding
            file already exists.

    Returns:
        tuple[int, int, int, list[str]]:
            - ``num_success`` — number of embeddings newly written.
            - ``num_errors``  — number of unrecoverable failures.
            - ``num_skipped`` — number of sizes skipped (existing file, OOM, etc.).
            - ``error_messages`` — list of human-readable error strings.
    """
    shape = image.shape[1:]  # [X, Y, Z]
    try:
        target_3d_sizes = generate_target_3d_sizes_mri(
            x_size=shape[0],
            y_size=shape[1],
            z_size=shape[2],
            original_spacing=effective_spacing,
            through_plane_axis=through_plane_axis,
        )
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Error generating target sizes: {str(e)}"]

    num_success = num_errors = num_skipped = 0
    error_messages = []

    for target_size in target_3d_sizes:
        success, message, _ = encode_and_save(
            image=image,
            out_filename_base=out_filename_base,
            target_size=target_size,
            autoencoder=autoencoder,
            device=device,
            num_splits=num_splits,
            skip_existing=skip_existing,
        )
        if success:
            if "Already exists" in message:
                num_skipped += 1
            else:
                num_success += 1
        else:
            # Silently skip sizes that are too large — either caught by the
            # static GPU memory guard or by a runtime OOM during encoding.
            if "too large for num_splits" in message or "OOM" in message:
                num_skipped += 1
            else:
                num_errors += 1
                error_messages.append(f"{image_rel_path} @ {target_size}: {message}")

    logger.debug("%s: encoded %d, skipped %d, errors %d", image_rel_path, num_success, num_skipped, num_errors)
    return num_success, num_errors, num_skipped, error_messages


def process_single_image(image_full_path, image_rel_path, output_root, transforms,
                         autoencoder, device, num_splits, skip_existing=True):
    """
    End-to-end pipeline for a single MRI file: preprocess → size generation → encode.

    Steps:

    1. Load and preprocess the image with ``transforms``.
    2. Extract voxel spacing from the RAS-reoriented MetaTensor affine.
    3. Identify the through-plane axis via ``identify_throughplane_axis``.
    4. Delegate to ``encode_all_target_sizes``, which generates all valid
       target 3-D sizes (enforcing equal in-plane dims and coarser
       through-plane spacing) and calls ``encode_and_save`` for each.

    Output files are placed under ``output_root`` mirroring the relative
    directory structure of ``image_rel_path``, with the ``.nii[.gz]`` extension
    replaced by ``_{x}x{y}x{z}_emb.nii.gz``.

    Args:
        image_full_path (str): Absolute path to the source NIfTI file.
        image_rel_path (str): Relative path used for logging and to determine
            the output sub-directory structure.
        output_root (str): Root directory under which all embeddings are saved.
        transforms (monai.transforms.Compose): Preprocessing pipeline, e.g. from
            ``create_preprocessing_transforms()``.
        autoencoder (torch.nn.Module): Loaded autoencoder in eval mode.
        device (torch.device): Encoding device.
        num_splits (int): Tensor-parallel split count for the GPU memory guard.
        skip_existing (bool): If ``True``, skip target sizes whose embedding
            file already exists.

    Returns:
        tuple[int, int, int, list[str]]:
            - ``num_success`` — number of embeddings newly written.
            - ``num_errors``  — number of unrecoverable failures.
            - ``num_skipped`` — number of sizes skipped (existing file, OOM, etc.).
            - ``error_messages`` — list of human-readable error strings.
    """
    if not os.path.exists(image_full_path):
        return 0, 1, 0, [f"Image not found: {image_full_path}"]

    # Load, orient, normalise and extract spacing
    try:
        image, original_spacing = load_and_get_spacing(
            image_full_path, image_rel_path, transforms
        )
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Preprocessing error: {str(e)}"]

    original_shape = image.shape[1:]  # [X, Y, Z]
    logger.debug("%s: shape=%s, spacing=%s", image_rel_path, list(original_shape), original_spacing)
    
    # Identify through-plane axis
    try:
        through_plane_axis = identify_throughplane_axis(
            x_size=original_shape[0],
            y_size=original_shape[1],
            z_size=original_shape[2],
            original_spacing=original_spacing,
            image_rel_path=image_rel_path,
        )
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Error identifying through-plane axis: {str(e)}"]

    # Build output base path (mirrors relative directory structure)
    rel_dir = os.path.dirname(image_rel_path)
    filename = os.path.basename(image_rel_path).replace('.nii.gz', '').replace('.nii', '')
    out_filename_base = os.path.join(output_root, rel_dir, filename)

    # Generate target sizes and encode; delegate to shared helper
    return encode_all_target_sizes(
        image=image,
        effective_spacing=original_spacing,
        through_plane_axis=through_plane_axis,
        image_rel_path=image_rel_path,
        out_filename_base=out_filename_base,
        autoencoder=autoencoder,
        device=device,
        num_splits=num_splits,
        skip_existing=skip_existing,
    )
