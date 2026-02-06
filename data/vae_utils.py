#!/usr/bin/env python3
"""
Shared utilities for VAE encoding across different datasets.

Reusable functions extracted from MR_Rate/3_apply_vae.py to avoid duplication.
"""

import os
import sys
import json
import torch
import nibabel as nib
import numpy as np
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
POSSIBLE_SIZES = [128, 256, 384, 512]
MIN_SIZE = 32


def download_autoencoder_model(local_rank=0):
    """Download the autoencoder model if not already present (only on rank 0)."""
    if local_rank == 0:
        if os.path.exists(AUTOENCODER_PATH):
            print(f"Autoencoder model already exists at {AUTOENCODER_PATH}")
            return AUTOENCODER_PATH
        
        print("Downloading autoencoder_v1.pt model...")
        # Download using rflow-ct which includes autoencoder_v1.pt
        download_model_data(generate_version="rflow-ct", root_dir=PROJECT_ROOT, model_only=True)
        
        if not os.path.exists(AUTOENCODER_PATH):
            raise FileNotFoundError(f"Failed to download autoencoder model to {AUTOENCODER_PATH}")
        
        print(f"✓ Autoencoder v1 model downloaded to {AUTOENCODER_PATH}")
    
    return AUTOENCODER_PATH


def load_autoencoder(device, num_splits=1):
    """
    Load the trained autoencoder model using config file (same as inference_tutorial.ipynb).
    
    Args:
        device: torch device
        num_splits: number of splits for tensor parallelism (for large images)
    
    Returns:
        Loaded autoencoder model in eval mode
    """
    print(f"Loading autoencoder from {AUTOENCODER_PATH}...")
    print(f"Using network config: {NETWORK_CONFIG}")
    
    # Load network configuration
    with open(NETWORK_CONFIG, 'r') as f:
        network_config = json.load(f)
    
    # Override num_splits if specified
    if num_splits is not None:
        network_config["autoencoder_def"]["num_splits"] = num_splits
        print(f"  Overriding num_splits to {num_splits}")
    
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
    
    print(f"✓ Autoencoder loaded successfully")
    print(f"  Architecture: {autoencoder.__class__.__name__}")
    
    return autoencoder


def create_preprocessing_transforms():
    """
    Create MRI preprocessing transforms.
    
    Based on: diffusion_model_scale_up/data/1_resample_128_v1_1.py
    - Load image
    - Ensure channel first
    - Orient to RAS
    - Intensity normalization (0-99.5 percentile to [0, 1])
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


def is_valid_image_size_for_splits_80g_a100(num_splits, image_size):
    """
    Check if image size is valid for given number of splits (GPU memory constraint).
    
    From 1_resample_128_v1_1.py
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
    Given the current size, return a list of possible downsample sizes.
    
    From 1_resample_128_v1_1.py: get_downsample_sizes()
    
    Args:
        current_size: Current dimension size
        possible_sizes: List of candidate sizes (e.g., [128, 256, 384, 512])
        
    Returns:
        List of valid downsample sizes up to and including current_size
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


def generate_target_3d_sizes_mri(x_size, y_size, z_size, original_spacing, possible_sizes=POSSIBLE_SIZES):
    """
    Generate all possible 3D target sizes for MRI with constraint:
    Two dimensions must be equal.
    Valid patterns: [x, x, z], [x, z, x], [z, x, x]
    
    The choice of which dimensions should be equal is based on:
    1. Spacing similarity: if two spacings are very close, those dimensions are kept equal
    2. Size similarity: if spacings are all close, use the two dimensions with closest sizes
    
    Modified from 1_resample_128_v1_1.py to enforce dimensional constraints.
    
    Args:
        x_size, y_size, z_size: Original image dimensions
        original_spacing: List of [x_spacing, y_spacing, z_spacing] in mm
        possible_sizes: List of candidate sizes
        
    Returns:
        List of target 3D size tuples [(x1,y1,z1), (x2,y2,z2), ...]
    """
    # Determine which two dimensions should be kept equal based on spacing
    # Calculate pairwise spacing ratios (smaller/larger to get value <= 1)
    sp_x, sp_y, sp_z = original_spacing[0], original_spacing[1], original_spacing[2]
    
    ratio_xy = min(sp_x, sp_y) / max(sp_x, sp_y) if max(sp_x, sp_y) > 0 else 1.0
    ratio_xz = min(sp_x, sp_z) / max(sp_x, sp_z) if max(sp_x, sp_z) > 0 else 1.0
    ratio_yz = min(sp_y, sp_z) / max(sp_y, sp_z) if max(sp_y, sp_z) > 0 else 1.0
    
    # Threshold for "similar spacing" (e.g., ratio > 0.8 means spacings are within 25% of each other)
    SPACING_SIMILARITY_THRESHOLD = 0.8
    
    # Determine which pair has most similar spacing
    if ratio_xy > max(ratio_xz, ratio_yz) and ratio_xy > SPACING_SIMILARITY_THRESHOLD:
        # X and Y have similar spacing -> keep them equal
        paired_dims = ('x', 'y')
        single_dim = 'z'
        paired_size = min(x_size, y_size)
        single_size = z_size
    elif ratio_xz > max(ratio_xy, ratio_yz) and ratio_xz > SPACING_SIMILARITY_THRESHOLD:
        # X and Z have similar spacing -> keep them equal
        paired_dims = ('x', 'z')
        single_dim = 'y'
        paired_size = min(x_size, z_size)
        single_size = y_size
    elif ratio_yz > max(ratio_xy, ratio_xz) and ratio_yz > SPACING_SIMILARITY_THRESHOLD:
        # Y and Z have similar spacing -> keep them equal
        paired_dims = ('y', 'z')
        single_dim = 'x'
        paired_size = min(y_size, z_size)
        single_size = x_size
    else:
        # All spacings are similar or all different - use dimension sizes instead
        # Find the two dimensions with most similar sizes
        size_diff_xy = abs(x_size - y_size)
        size_diff_xz = abs(x_size - z_size)
        size_diff_yz = abs(y_size - z_size)
        
        if size_diff_xy <= min(size_diff_xz, size_diff_yz):
            # X and Y have most similar sizes
            paired_dims = ('x', 'y')
            single_dim = 'z'
            paired_size = min(x_size, y_size)
            single_size = z_size
        elif size_diff_xz <= min(size_diff_xy, size_diff_yz):
            # X and Z have most similar sizes
            paired_dims = ('x', 'z')
            single_dim = 'y'
            paired_size = min(x_size, z_size)
            single_size = y_size
        else:
            # Y and Z have most similar sizes
            paired_dims = ('y', 'z')
            single_dim = 'x'
            paired_size = min(y_size, z_size)
            single_size = x_size
    
    # Create dimension map
    dimension_map = {}
    for dim in paired_dims:
        dimension_map[dim] = paired_size
    dimension_map[single_dim] = single_size
    
    # Get downsample sizes for paired and single dimensions
    paired_targets = get_downsample_sizes(paired_size, possible_sizes)
    single_targets = get_downsample_sizes(single_size, possible_sizes)
    
    # Ensure at least one value per dimension
    if not paired_targets:
        paired_targets = [min(possible_sizes)]
    if not single_targets:
        single_targets = [min(possible_sizes)]
    
    # Generate target sizes with constraint: paired dims are equal
    target_3d_sizes_set = set()
    
    for paired_val in paired_targets:
        for single_val in single_targets:
            # Build size tuple based on which dimensions are paired
            size_dict = {}
            for dim in paired_dims:
                size_dict[dim] = paired_val
            size_dict[single_dim] = single_val
            
            # Add to set (in x, y, z order)
            target_3d_sizes_set.add((size_dict['x'], size_dict['y'], size_dict['z']))
    
    # Convert to sorted list
    target_3d_sizes = sorted(target_3d_sizes_set)
    
    return target_3d_sizes


def check_spacing_constraint(original_shape, original_spacing, target_size):
    """
    Check if the resampled spacing satisfies the constraint: x-spacing <= z-spacing.
    
    For dimensions in format [x, x, z] or [x, z, x] or [z, x, x],
    where two dimensions are equal (x) and one is different (z),
    we require that the spacing of x dimensions <= spacing of z dimension.
    
    Args:
        original_shape: [dim0, dim1, dim2] original dimensions
        original_spacing: [sp0, sp1, sp2] original spacing in mm
        target_size: [dim0, dim1, dim2] target dimensions
        
    Returns:
        True if constraint is satisfied, False otherwise
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


def encode_and_save(image, out_filename_base, target_size, autoencoder, device, num_splits, skip_existing=True):
    """
    Downsample image, encode with VAE, and save embedding.
    
    Based on: downsample_and_save() in 1_resample_128_v1_1.py
    
    Args:
        image: Preprocessed image tensor with metadata
        out_filename_base: Base path for output file
        target_size: Target 3D size [x, y, z]
        autoencoder: Trained VAE model
        device: torch device
        num_splits: Number of splits for tensor parallelism
        skip_existing: Whether to skip if output already exists
        
    Returns:
        tuple: (success: bool, message: str, output_path: str or None)
    """
    x_target, y_target, z_target = target_size
    
    # Define save path
    save_path = f"{out_filename_base}_{x_target}x{y_target}x{z_target}_emb.nii.gz"
    
    # Check if output already exists AND verify it's not broken
    if skip_existing and os.path.exists(save_path):
        try:
            # Try to load the file to verify it's not corrupted
            test_nii = nib.load(save_path)
            test_data = test_nii.get_fdata()
            # If we can load it successfully, it's valid - skip it
            return True, "Already exists (verified)", save_path
        except Exception as e:
            # File exists but is corrupted - reprocess it
            pass  # Continue to reprocess
    
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


def process_single_image(image_full_path, image_rel_path, output_root, transforms, 
                        autoencoder, device, num_splits, skip_existing=True):
    """
    Process a single MR image: preprocess, generate target sizes, encode with VAE.
    
    This is a shared function used by both MR_Rate and others dataset scripts.
    
    Args:
        image_full_path: Full path to the image file
        image_rel_path: Relative path (for output structure and logging)
        output_root: Root directory for output embeddings
        transforms: MONAI preprocessing transforms
        autoencoder: Trained VAE model
        device: torch device
        num_splits: Number of splits for tensor parallelism
        skip_existing: Whether to skip existing embeddings
        
    Returns:
        tuple: (num_success, num_errors, num_skipped, error_messages_list)
    """
    # Check if image exists
    if not os.path.exists(image_full_path):
        return 0, 1, 0, [f"Image not found: {image_full_path}"]
    
    # Preprocess image
    try:
        data_dict = {"image": image_full_path}
        preprocessed = transforms(data_dict)
        image = preprocessed["image"]
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Preprocessing error: {str(e)}"]
    
    # Get original size and spacing
    original_shape = image.shape[1:]  # [H, W, D]
    
    # Extract spacing from NIfTI header
    try:
        nii_img = nib.load(image_full_path)
        if hasattr(nii_img, 'header'):
            pixdim = nii_img.header.get_zooms()
            if len(pixdim) >= 3:
                original_spacing = [float(pixdim[i]) for i in range(3)]
            else:
                original_spacing = [1.0, 1.0, 1.0]
        else:
            original_spacing = [1.0, 1.0, 1.0]
    except Exception as e:
        print(f"[WARNING] {image_rel_path}: Could not extract spacing, using [1.0, 1.0, 1.0]")
        original_spacing = [1.0, 1.0, 1.0]
    
    # Debug output
    print(f"[DEBUG] {image_rel_path}: original shape = {original_shape}, spacing = {original_spacing}")
    
    # Generate target sizes
    try:
        target_3d_sizes = generate_target_3d_sizes_mri(
            x_size=original_shape[0],
            y_size=original_shape[1],
            z_size=original_shape[2],
            original_spacing=original_spacing,
            possible_sizes=POSSIBLE_SIZES
        )
    except Exception as e:
        return 0, 1, 0, [f"{image_rel_path}: Error generating target sizes: {str(e)}"]
    
    # Define output base path (preserve original relative directory structure)
    # Remove .nii.gz extension from filename
    rel_dir = os.path.dirname(image_rel_path)
    filename = os.path.basename(image_rel_path).replace('.nii.gz', '').replace('.nii', '')
    out_filename_base = os.path.join(output_root, rel_dir, filename)
    
    # Encode and save for each target size
    num_success = 0
    num_errors = 0
    num_skipped = 0
    error_messages = []
    
    for target_size in target_3d_sizes:
        # Check spacing constraint: x-spacing <= z-spacing
        if not check_spacing_constraint(original_shape, original_spacing, target_size):
            # Skip this target size - doesn't satisfy spacing constraint
            num_skipped += 1
            continue
        
        success, message, output_path = encode_and_save(
            image=image,
            out_filename_base=out_filename_base,
            target_size=target_size,
            autoencoder=autoencoder,
            device=device,
            num_splits=num_splits,
            skip_existing=skip_existing
        )
        
        if success:
            if "Already exists" in message:
                num_skipped += 1
            else:
                num_success += 1
        else:
            # Silently skip images that are too large (expected cases)
            if "too large for num_splits" in message:
                num_skipped += 1
            else:
                num_errors += 1
                error_messages.append(f"{image_rel_path} @ {target_size}: {message}")
    
    return num_success, num_errors, num_skipped, error_messages
