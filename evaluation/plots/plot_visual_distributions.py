import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_visual_portions(json_path):
    """
    Load visual portion data from JSON file.

    Args:
        json_path: Path to JSON file containing image_filename -> visual_portion mapping

    Returns:
        numpy.ndarray: Array of visual portion values (0.0 to 1.0)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract values (visual portions) as numpy array
    visual_portions = np.array(list(data.values()))
    return visual_portions


def format_dataset_name(filename):
    """
    Convert filename to properly formatted dataset name.

    Args:
        filename: JSON filename (e.g., 'finreport_visual_portions.json')

    Returns:
        str: Formatted dataset name (e.g., 'FinReport')
    """
    # Remove _visual_portions.json suffix
    name = filename.replace('_visual_portions.json', '')

    # Special case mappings
    name_map = {
        'finreport': 'FinReport',
        'finqa': 'FinQA',
        'convfinqa': 'ConvFinQA',
        'tatdqa': 'TATDQA',
        'vqaonbd': 'VQAonBD',
        'finslides': 'FinSlides',
        'techreport': 'TechReport',
    }

    return name_map.get(name.lower(), name.title())


def create_combined_histogram(datasets_data, output_path=None):
    """
    Create a combined histogram showing all datasets as smooth density curves.

    Args:
        datasets_data: List of tuples (dataset_name, data_array)
        output_path: Optional path to save figure
    """
    # Create figure
    plt.figure(figsize=(14, 8))

    # Define highly distinct color palette
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Grey
        '#00CED1',  # Dark Turquoise
        '#FF1493',  # Deep Pink
    ]

    # Cycle through colors if more datasets than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(datasets_data))]

    # Define bins for histogram
    bins = np.linspace(0, 1, 21)  # 20 bins from 0.0 to 1.0
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute proportions for all datasets
    proportions_list = []
    labels = []
    for dataset_name, data in datasets_data:
        # Compute histogram
        counts, _ = np.histogram(data, bins=bins)

        # Normalize to proportions (sum = 1.0)
        # Divide by total documents to get fraction in each bin
        proportions = counts / (11*len(data)) if len(data) > 0 else counts

        proportions_list.append(proportions)
        labels.append(dataset_name)

    # Create stacked area chart
    plt.stackplot(bin_centers, *proportions_list, labels=labels, colors=colors, alpha=0.8)

    # Styling
    plt.ylabel('Proportion of Visual Content (%)', fontsize=25, fontweight='bold')
    plt.xlabel('Visual Content Density', fontsize=25, fontweight='bold')

    # Add legend with smaller font and better positioning
    plt.legend(loc='upper right', fontsize=28, frameon=False,
              fancybox=False, shadow=True, ncol=3)

    # Set x-axis limits
    plt.xlim(0, 1)
    plt.ylim(0, None)  # Auto-scale y-axis based on actual proportions
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add grid
    plt.grid(False, alpha=0.3, linestyle='', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()

    plt.close()


def process_folder(folder_path, output_dir=None):
    """
    Process all JSON files in a folder and create combined histogram.

    Args:
        folder_path: Path to folder containing JSON files
        output_dir: Optional directory to save figures
    """
    folder = Path(folder_path)

    # Find all JSON files matching the pattern
    json_files = sorted(folder.glob('*_visual_portions.json'))

    if not json_files:
        print(f"No *_visual_portions.json files found in {folder_path}")
        return

    print(f"Found {len(json_files)} dataset(s):")
    for json_file in json_files:
        print(f"  - {json_file.name}")
    print()

    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all datasets
    datasets_data = []
    for json_file in json_files:
        # Load data
        data = load_visual_portions(json_file)

        # Format dataset name
        dataset_name = format_dataset_name(json_file.name)

        print(f"Loaded {dataset_name}: {len(data)} files")

        datasets_data.append((dataset_name, data))

    # Create combined histogram
    print(f"\nCreating combined histogram for {len(datasets_data)} dataset(s)...")

    output_path = None
    if output_dir:
        output_path = Path(output_dir) / 'visual_distributions_combined.png'

    create_combined_histogram(datasets_data, output_path)

    print(f"\nCompleted processing {len(json_files)} dataset(s)!")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Create histograms of visual content distributions from JSON files.'
    )

    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to folder containing *_visual_portions.json files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./figures/',
        help='Output directory to save figures (default: ./figures/)'
    )

    args = parser.parse_args()

    # Process folder
    process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
