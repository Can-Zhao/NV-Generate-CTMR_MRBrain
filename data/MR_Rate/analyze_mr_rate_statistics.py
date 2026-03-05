#!/usr/bin/env python3
"""
Script to analyze MR-RATE dataset statistics and create scatter plots.

Reads the JSON file with 'dim' and 'spacing' information and:
1. Computes statistics for each modality
2. Creates scatter plots showing relationships between dimensions, spacing, and FOV

This script should be run after add_dim_spacing_to_json.py
"""

import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import re

# Configuration
JSON_FILE = "/lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_MR-RATE_brain_mask_pairs.json"
METADATA_ROOT = None


def identify_axes(affine, spacing):
    """
    Identify in-plane and through-plane axes based on affine matrix and spacing.
    
    The through-plane axis is typically the one with the largest spacing (slice thickness).
    The in-plane axes are the other two axes.
    
    Args:
        affine (np.ndarray): 4x4 affine matrix (not used but kept for compatibility)
        spacing (np.ndarray): Voxel spacing [x, y, z]
        
    Returns:
        tuple: (in_plane_axes, through_plane_axis)
            in_plane_axes: list of two axis indices [0, 1, 2]
            through_plane_axis: single axis index
    """
    # Find the axis with maximum spacing (through-plane)
    through_plane_axis = np.argmax(spacing)
    
    # The other two axes are in-plane
    in_plane_axes = [i for i in range(3) if i != through_plane_axis]
    
    return in_plane_axes, through_plane_axis


def compute_statistics(values):
    """
    Compute statistics for a list of values.
    
    Args:
        values (list): List of numeric values
        
    Returns:
        dict: Dictionary with statistics (mean, std, min, max, median, q25, q75)
    """
    if not values:
        return {}
    
    values = np.array(values)
    
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
    }


def extract_modality_and_plane(modality_key):
    """
    Extract base modality and acquisition plane from modality key.
    Strips numeric suffixes (e.g., "-2", "-3") to group variants together.
    
    Examples:
        'flair-raw-axi' -> ('flair-raw-axi', 'axi')
        'flair-raw-axi-2' -> ('flair-raw-axi', 'axi')
        't1w-raw-sag-3' -> ('t1w-raw-sag', 'sag')
        'swi-raw-cor' -> ('swi-raw-cor', 'cor')
    
    Args:
        modality_key (str): Modality key string
        
    Returns:
        tuple: (base_modality_key, plane) or (modality_key, None) if plane not found
    """
    # First, remove any trailing numeric suffixes (e.g., "-2", "-10", "-25")
    # This groups variants like "flair-raw-axi-2" with "flair-raw-axi"
    base_modality_key = re.sub(r'-\d+$', '', modality_key)
    
    # Try to extract plane (axi, sag, cor, obl)
    for plane in ['axi', 'sag', 'cor', 'obl']:
        if f'-{plane}' in base_modality_key:
            return (base_modality_key, plane)
    
    return (modality_key, None)


def get_through_plane_axis_name(plane):
    """
    Map acquisition plane to through-plane axis name.
    
    Args:
        plane (str): Acquisition plane ('sag', 'cor', 'axi', or None)
        
    Returns:
        str: Axis name ('x-axis', 'y-axis', 'z-axis', or 'unknown')
    """
    plane_to_axis = {
        'sag': 'x-axis',  # Sagittal: through-plane is X-axis
        'cor': 'y-axis',  # Coronal: through-plane is Y-axis
        'axi': 'z-axis',  # Axial: through-plane is Z-axis
    }
    return plane_to_axis.get(plane, 'unknown')


def analyze_image_from_json(pair):
    """
    Analyze an image entry from JSON to extract dimensional and spacing information.
    
    Args:
        pair (dict): Dictionary containing image information with 'dim' and 'spacing' keys
        
    Returns:
        dict: Dictionary containing analysis results or None if error
    """
    if 'dim' not in pair or 'spacing' not in pair:
        return None
    
    try:
        dims = np.array(pair['dim'])
        spacing = np.array(pair['spacing'])
        
        # Get modality key to determine plane
        modality_key = pair.get('modality_key', '')
        _, plane = extract_modality_and_plane(modality_key)
        
        # Identify in-plane and through-plane axes
        in_plane_axes, through_plane_axis = identify_axes(None, spacing)
        
        # Get through-plane axis name based on acquisition plane
        through_plane_axis_name = get_through_plane_axis_name(plane)
        
        # Calculate in-plane dimensions
        in_plane_dims = [dims[in_plane_axes[0]], dims[in_plane_axes[1]]]
        in_plane_spacing = [spacing[in_plane_axes[0]], spacing[in_plane_axes[1]]]
        
        # Calculate through-plane dimension and spacing
        through_plane_dim = dims[through_plane_axis]
        through_plane_spacing = spacing[through_plane_axis]
        
        # Check if through-plane spacing is smaller than in-plane spacing (unusual case)
        avg_in_plane_spacing = (in_plane_spacing[0] + in_plane_spacing[1]) / 2.0
        if through_plane_spacing < avg_in_plane_spacing:
            image_path = pair.get('image', 'Unknown')
            print(f"Warning: Through-plane spacing ({through_plane_spacing:.4f} mm) is smaller than average in-plane spacing ({avg_in_plane_spacing:.4f} mm) for {image_path}")
            print(f"  This is unusual - typically slice thickness should be larger than in-plane resolution")
        
        # Calculate in-plane FOV (field of view) = dimension * voxel_size
        in_plane_fov_0 = in_plane_dims[0] * in_plane_spacing[0]
        in_plane_fov_1 = in_plane_dims[1] * in_plane_spacing[1]
        in_plane_fov = [in_plane_fov_0, in_plane_fov_1]
        
        # Calculate through-plane FOV
        through_plane_fov = through_plane_dim * through_plane_spacing
        
        return {
            'dims': dims.tolist(),
            'spacing': spacing.tolist(),
            'in_plane_axes': in_plane_axes,
            'through_plane_axis': int(through_plane_axis),
            'through_plane_axis_name': through_plane_axis_name,
            'in_plane_dims': in_plane_dims,
            'in_plane_spacing': in_plane_spacing,
            'in_plane_fov': in_plane_fov,
            'in_plane_fov_0': float(in_plane_fov_0),
            'in_plane_fov_1': float(in_plane_fov_1),
            'through_plane_dim': int(through_plane_dim),
            'through_plane_spacing': float(through_plane_spacing),
            'through_plane_fov': float(through_plane_fov),
        }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


def create_dot_plots(modality_data, output_dir='./jsons'):
    """
    Create scatter plots showing relationships for each modality and plane combination.
    
    Args:
        modality_data (dict): Dictionary mapping modality_key -> list of analysis results
        output_dir (str): Directory to save plots
    """
    # Group by base modality and plane
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    for modality_key, data_list in modality_data.items():
        base_modality, plane = extract_modality_and_plane(modality_key)
        if plane:
            grouped_data[base_modality][plane].extend(data_list)
    
    # Create plots for each base modality
    for base_modality in sorted(grouped_data.keys()):
        planes_data = grouped_data[base_modality]
        
        # Process if we have any valid planes (axi, sag, cor, or obl)
        valid_planes = [p for p in ['axi', 'sag', 'cor', 'obl'] if p in planes_data]
        if not valid_planes:
            continue
        
        # Prepare data for plotting
        plot_data = {}
        for plane in valid_planes:
            data_list = planes_data[plane]
            plot_data[plane] = {
                'in_plane_dim_0': [d['in_plane_dims'][0] for d in data_list],
                'in_plane_dim_1': [d['in_plane_dims'][1] for d in data_list],
                'in_plane_spacing_0': [d['in_plane_spacing'][0] for d in data_list],
                'in_plane_spacing_1': [d['in_plane_spacing'][1] for d in data_list],
                'in_plane_fov_0': [d['in_plane_fov_0'] for d in data_list],
                'in_plane_fov_1': [d['in_plane_fov_1'] for d in data_list],
                'through_plane_dim': [d['through_plane_dim'] for d in data_list],
                'through_plane_spacing': [d['through_plane_spacing'] for d in data_list],
                'through_plane_fov': [d['through_plane_fov'] for d in data_list],
            }
        
        # Create figure with 3 rows x 2 columns (6 plots total)
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(f'{base_modality.upper()} - Relationships by Acquisition Plane', fontsize=16, fontweight='bold')
        
        # Colors for each plane
        colors = {'axi': '#1f77b4', 'sag': '#ff7f0e', 'cor': '#2ca02c', 'obl': '#9467bd'}
        plane_labels = {'axi': 'Axial', 'sag': 'Sagittal', 'cor': 'Coronal', 'obl': 'Oblique'}
        
        # Row 1: Dimensions
        # Plot 1: In-Plane Dimension 0 vs In-Plane Dimension 1
        ax = axes[0, 0]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_dim_0']
            y_values = plot_data[plane]['in_plane_dim_1']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane Dimension 0 (pixels)')
        ax.set_ylabel('In-Plane Dimension 1 (pixels)')
        ax.set_title('Dimensions: In-Plane 0 vs In-Plane 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Plot 2: In-Plane Dimension 0 vs Through-Plane Dimension
        ax = axes[0, 1]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_dim_0']
            y_values = plot_data[plane]['through_plane_dim']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane Dimension 0 (pixels)')
        ax.set_ylabel('Through-Plane Dimension (slices)')
        ax.set_title('Dimensions: In-Plane 0 vs Through-Plane')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
        
        # Row 2: Spacing
        # Plot 3: In-Plane Spacing 0 vs In-Plane Spacing 1
        ax = axes[1, 0]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_spacing_0']
            y_values = plot_data[plane]['in_plane_spacing_1']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane Spacing 0 (mm)')
        ax.set_ylabel('In-Plane Spacing 1 (mm)')
        ax.set_title('Spacing: In-Plane 0 vs In-Plane 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Plot 4: In-Plane Spacing 0 vs Through-Plane Spacing
        ax = axes[1, 1]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_spacing_0']
            y_values = plot_data[plane]['through_plane_spacing']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane Spacing 0 (mm)')
        ax.set_ylabel('Through-Plane Spacing (mm)')
        ax.set_title('Spacing: In-Plane 0 vs Through-Plane')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
        
        # Row 3: FOV
        # Plot 5: In-Plane FOV 0 vs In-Plane FOV 1
        ax = axes[2, 0]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_fov_0']
            y_values = plot_data[plane]['in_plane_fov_1']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane FOV 0 (mm)')
        ax.set_ylabel('In-Plane FOV 1 (mm)')
        ax.set_title('FOV: In-Plane 0 vs In-Plane 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Plot 6: In-Plane FOV 0 vs Through-Plane FOV
        ax = axes[2, 1]
        for plane in valid_planes:
            x_values = plot_data[plane]['in_plane_fov_0']
            y_values = plot_data[plane]['through_plane_fov']
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color=colors[plane], 
                      label=plane_labels[plane], edgecolors='none')
        ax.set_xlabel('In-Plane FOV 0 (mm)')
        ax.set_ylabel('Through-Plane FOV (mm)')
        ax.set_title('FOV: In-Plane 0 vs Through-Plane')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{base_modality}_distribution_dotplots.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved dot plot: {output_file}")
        plt.close()


def create_modality_count_barplot(modality_stats, output_dir='./jsons'):
    """
    Create a bar plot showing the count of images for each modality.
    
    Args:
        modality_stats (dict): Dictionary mapping modality -> statistics dict
        output_dir (str): Directory to save the plot
    """
    # Extract modality names and counts
    modalities = []
    counts = []
    
    for modality in sorted(modality_stats.keys()):
        stats = modality_stats[modality]
        modalities.append(modality)
        counts.append(stats['count'])
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    bars = ax.bar(range(len(modalities)), counts, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by modality type (flair, t1w, t2w, swi, mra)
    color_map = {
        'flair': '#1f77b4',
        't1w': '#ff7f0e',
        't2w': '#2ca02c',
        'swi': '#d62728',
        'mra': '#9467bd',
    }
    
    for i, modality in enumerate(modalities):
        # Determine color based on modality prefix
        color = '#808080'  # Default gray
        for prefix, bar_color in color_map.items():
            if modality.startswith(prefix):
                color = bar_color
                break
        bars[i].set_color(color)
    
    # Set labels and title
    ax.set_xlabel('Modality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Image Count by Modality', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(modalities, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'modality_count_barplot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved bar plot: {output_file}")
    plt.close()


def analyze_dataset():
    """
    Main function to analyze the MR-RATE dataset.
    """
    # Load the JSON file
    print("Loading JSON file...")
    with open(JSON_FILE, 'r') as f:
        image_pairs = json.load(f)
    
    print(f"Found {len(image_pairs)} image-mask pairs")
    print("=" * 80)
    
    # Group by modality
    modality_data = defaultdict(list)
    
    # Process each image
    print("Processing images...")
    for idx, pair in enumerate(image_pairs):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(image_pairs)} images...")
        
        modality_key = pair.get('modality_key', 'Unknown')
        
        # Analyze the image from JSON data
        analysis = analyze_image_from_json(pair)
        
        if analysis:
            analysis['modality'] = modality_key
            analysis['subject_id'] = pair.get('subject_id', '')
            modality_data[modality_key].append(analysis)
    
    print(f"\nProcessed {sum(len(v) for v in modality_data.values())} images")
    print("=" * 80)
    
    # Group modalities by base form (without numeric suffixes) before computing statistics
    grouped_modality_data = defaultdict(list)
    for modality_key, data_list in modality_data.items():
        base_modality, _ = extract_modality_and_plane(modality_key)
        grouped_modality_data[base_modality].extend(data_list)
    
    # Compute statistics for each base modality
    modality_stats = {}
    
    for base_modality in sorted(grouped_modality_data.keys()):
        data_list = grouped_modality_data[base_modality]
        
        if not data_list:
            continue
        
        # Extract values for statistics
        in_plane_dim_0 = [d['in_plane_dims'][0] for d in data_list]
        in_plane_dim_1 = [d['in_plane_dims'][1] for d in data_list]
        in_plane_spacing_0 = [d['in_plane_spacing'][0] for d in data_list]
        in_plane_spacing_1 = [d['in_plane_spacing'][1] for d in data_list]
        in_plane_fov_0 = [d['in_plane_fov_0'] for d in data_list]
        in_plane_fov_1 = [d['in_plane_fov_1'] for d in data_list]
        through_plane_dim = [d['through_plane_dim'] for d in data_list]
        through_plane_spacing = [d['through_plane_spacing'] for d in data_list]
        through_plane_fov = [d['through_plane_fov'] for d in data_list]
        
        # Get through-plane axis name (should be the same for all images in this modality)
        # Use the most common axis name, or the first one if all are the same
        through_plane_axis_names = [d.get('through_plane_axis_name', 'unknown') for d in data_list]
        most_common_axis = max(set(through_plane_axis_names), key=through_plane_axis_names.count) if through_plane_axis_names else 'unknown'
        
        # Compute statistics
        stats = {
            'modality': base_modality,
            'count': len(data_list),
            'in_plane_dim_0': compute_statistics(in_plane_dim_0),
            'in_plane_dim_1': compute_statistics(in_plane_dim_1),
            'in_plane_spacing_0': compute_statistics(in_plane_spacing_0),
            'in_plane_spacing_1': compute_statistics(in_plane_spacing_1),
            'in_plane_fov_0': compute_statistics(in_plane_fov_0),
            'in_plane_fov_1': compute_statistics(in_plane_fov_1),
            'through_plane_axis': most_common_axis,  # Store axis name (x-axis, y-axis, or z-axis)
            'through_plane_dim': compute_statistics(through_plane_dim),
            'through_plane_spacing': compute_statistics(through_plane_spacing),
            'through_plane_fov': compute_statistics(through_plane_fov),
        }
        
        modality_stats[base_modality] = stats
    
    # Summary by base modality (t1w, t2w, flair, swi, mra): aggregate counts
    modality_summary = defaultdict(int)
    for base_modality, stats in modality_stats.items():
        # base_modality is e.g. 'flair-raw-axi', 't1w-raw-sag' -> take prefix: flair, t1w, t2w, swi, mra
        prefix = base_modality.split('-')[0] if base_modality else 'unknown'
        modality_summary[prefix] += stats['count']
    modality_summary = dict(modality_summary)
    
    # Print summary by modality to console
    print("\n" + "=" * 80)
    print("=== Summary by modality (t1, t2, flair, etc.) ===")
    print("=" * 80)
    for mod in sorted(modality_summary.keys()):
        print(f"  {mod}: {modality_summary[mod]}")
    print(f"  TOTAL: {sum(modality_summary.values())}")
    print("=" * 80)
    
    # Save statistics to JSON file (include modality summary)
    output_file = "./jsons/mr_rate_dataset_statistics.json"
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"Saving statistics to {output_file}...")
    output_data = {
        "modality_summary": modality_summary,
        "modality_stats": modality_stats,
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Statistics saved successfully!")
    
    # Create dot plots
    print("\n" + "=" * 80)
    print("Creating dot plots...")
    create_dot_plots(modality_data, output_dir=output_dir)
    
    # Create bar plot for modality counts
    print("\n" + "=" * 80)
    print("Creating bar plot for modality counts...")
    create_modality_count_barplot(modality_stats, output_dir=output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_dataset()
