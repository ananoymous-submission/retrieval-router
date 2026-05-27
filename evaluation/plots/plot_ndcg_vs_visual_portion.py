import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed


# Dataset mapping
DATASET_HF_IDS = {
    'finreport': 'ananoymous/FinReport',
    'finslides': 'ananoymous/FinSlides',
    'finqa': 'ananoymous/FinQA',
    'convfinqa': 'ananoymous/ConvFinQA',
    'vqaonbd': 'ananoymous/VQAonBD',
    'tatdqa': 'ananoymous/TATDQA',
    'mp-docvqa': 'ananoymous/MP-DocVQA',
    'arxivqa': 'ananoymous/ArxivQA',
    'dude': 'ananoymous/DUDE',
    'sciqag': 'ananoymous/SciQAG',
    'wiki-ss': 'ananoymous/Wiki-ss',
}


def load_visual_portions_all(visual_portions_folder):
    """
    Load all visual portion JSON files and combine them.

    Returns:
        dict: Mapping from image_filename to visual_portion across all datasets
    """
    visual_portions = {}

    folder = Path(visual_portions_folder)
    for json_file in folder.glob('*_visual_portions.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            visual_portions.update(data)

    print(f"Loaded visual portions for {len(visual_portions)} images")
    return visual_portions


def _load_single_dataset(dataset_name, hf_id, hf_token):
    """
    Load a single HuggingFace dataset and extract query → image_filename mapping.

    Args:
        dataset_name: Name of the dataset
        hf_id: HuggingFace dataset ID
        hf_token: HuggingFace token

    Returns:
        tuple: (dataset_name, query_to_image dict, count) or (dataset_name, None, error_msg)
    """
    try:
        # Load all splits and concatenate
        dataset = load_dataset(hf_id, token=hf_token)
        all_splits = []
        for key in dataset.keys():
            # Select only needed columns (drop heavy image column)
            split = dataset[key].select_columns(['query', 'image_filename'])
            all_splits.append(split)
        combined = concatenate_datasets(all_splits)

        # Map query → image_filename
        query_to_image = {row['query']: row['image_filename'] for row in combined}

        return (dataset_name, query_to_image, len(combined))
    except Exception as e:
        return (dataset_name, None, str(e))


def load_query_to_image_mapping():
    """
    Load HuggingFace datasets in parallel and create query → image_filename mapping.

    Returns:
        dict: Mapping from query to image_filename
    """
    query_to_image = {}
    hf_token = os.getenv('HF_TOKEN')

    print("Loading all datasets in parallel...")

    # Load all datasets in parallel
    with ThreadPoolExecutor(max_workers=len(DATASET_HF_IDS)) as executor:
        futures = {
            executor.submit(_load_single_dataset, name, hf_id, hf_token): name
            for name, hf_id in DATASET_HF_IDS.items()
        }

        for future in as_completed(futures):
            dataset_name, result, info = future.result()
            if result is not None:
                query_to_image.update(result)
                print(f"  Loaded {dataset_name}: {info} queries")
            else:
                print(f"  Error loading {dataset_name}: {info}")

    print(f"\nTotal query-to-image mappings: {len(query_to_image)}")
    return query_to_image


def prepare_data(predictions_path, visual_portions_folder):
    """
    Prepare data for plotting by combining predictions with visual portions.

    Returns:
        pd.DataFrame: DataFrame with query, dataset, visual_portion, NDCG for each pipeline, and RAG-MIXER
    """
    # Load predictions
    print("\nLoading predictions...")
    predictions = pd.read_excel(predictions_path)
    print(f"Loaded {len(predictions)} predictions")

    # Load visual portions
    print("\nLoading visual portions...")
    visual_portions = load_visual_portions_all(visual_portions_folder)

    # Load query-to-image mapping
    print("\nLoading query-to-image mapping...")
    query_to_image = load_query_to_image_mapping()

    # Add visual portion to predictions
    print("\nMatching queries to visual portions...")
    visual_portion_list = []
    for _, row in predictions.iterrows():
        query = row['query']
        image_filename = query_to_image.get(query)

        if image_filename and image_filename in visual_portions:
            visual_portion_list.append(visual_portions[image_filename])
        else:
            visual_portion_list.append(None)

    predictions['visual_portion'] = visual_portion_list

    # Remove rows without visual portion
    before_count = len(predictions)
    predictions = predictions.dropna(subset=['visual_portion'])
    after_count = len(predictions)
    print(f"Removed {before_count - after_count} queries without visual portion")
    print(f"Remaining queries: {after_count}")

    return predictions


def bin_and_aggregate(predictions, bins=5):
    """
    Bin queries by visual portion and aggregate NDCG values.

    Args:
        predictions: DataFrame with visual_portion and NDCG columns
        bins: Number of bins or list of bin edges

    Returns:
        pd.DataFrame: Aggregated data with bin centers and mean NDCG for each pipeline
    """
    # Create bins explicitly from 0.0 to 1.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    # Create bins
    predictions['visual_bin'] = pd.cut(
        predictions['visual_portion'],
        bins=bin_edges,
        labels=False,
        include_lowest=True
    )

    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Aggregate NDCG for each pipeline by bin
    pipelines = ['TEXT-SINGLE', 'TEXT-MULTI', 'MULTIMODAL-SINGLE', 'MULTIMODAL-MULTI']

    aggregated_data = []
    for bin_idx in range(bins):
        bin_data = predictions[predictions['visual_bin'] == bin_idx]

        if len(bin_data) == 0:
            continue

        row = {'bin_center': bin_centers[bin_idx], 'count': len(bin_data)}

        # Add mean NDCG for each pipeline
        for pipeline in pipelines:
            col_name = f'{pipeline}_ndcg'
            row[pipeline] = bin_data[col_name].mean()

        # Add RAG-MIXER predicted NDCG
        row['RAG-MIXER'] = bin_data['predicted_ndcg'].mean()

        # Add Oracle NDCG
        row['Oracle'] = bin_data['oracle_ndcg'].mean()

        aggregated_data.append(row)

    return pd.DataFrame(aggregated_data)


def plot_ndcg_vs_visual_portion(aggregated_data, output_path=None):
    """
    Create plot of NDCG vs Visual Portion for all pipelines.

    Args:
        aggregated_data: DataFrame with bin_center and mean NDCG for each pipeline
        output_path: Optional path to save the figure
    """
    pipelines = ['TEXT-SINGLE', 'TEXT-MULTI', 'MULTIMODAL-SINGLE', 'MULTIMODAL-MULTI']
    # Hierarchical colors: TEXT = yellow/brown, MULTIMODAL = red/orange
    colors = ['#F4A460', '#8B4513', '#FF6B35', '#C41E3A']  # Sandy Brown, Saddle Brown, Orange-Red, Cardinal Red
    linestyles = ['-', '-', '--', '--']
    markers = ['o', 's', 'o', 's']

    plt.figure(figsize=(10, 7))

    # Plot each pipeline
    for pipeline, color, linestyle, marker in zip(pipelines, colors, linestyles, markers):
        # Rename for legend
        label_map = {
            'TEXT-SINGLE': 'Text Dense',
            'TEXT-MULTI': 'Text Late',
            'MULTIMODAL-SINGLE': 'Multimodal Dense',
            'MULTIMODAL-MULTI': 'Multimodal Late'
        }
        label = label_map.get(pipeline, pipeline)

        plt.plot(
            aggregated_data['bin_center'],
            aggregated_data[pipeline],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=8,
            linewidth=2
        )

    # Plot RAG-MIXER (blue)
    plt.plot(
        aggregated_data['bin_center'],
        aggregated_data['RAG-MIXER'],
        label='RAG-MIXER',
        color='#1E90FF',  # Dodger Blue
        linestyle='-',
        marker='D',
        markersize=8,
        linewidth=2.5
    )

    # Plot Oracle (green)
    plt.plot(
        aggregated_data['bin_center'],
        aggregated_data['Oracle'],
        label='Oracle',
        color='#009E73',  # Forest Green
        linestyle='--',
        marker='*',
        markersize=10,
        linewidth=2.5
    )

    # Styling
    plt.xlabel('Visual Content Density', fontsize=24, fontweight='bold')
    plt.ylabel('NDCG@5', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Legend above figure, 3 columns (creates 2 rows for 6 items), bigger markers
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        fontsize=20,
        frameon=False,
        handlelength=2.5,
        markerscale=2.0
    )
    plt.xlim(-0.05, 1.05)
    # Adaptive y-axis limits

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot NDCG vs Visual Portion for base pipelines and RAG-MIXER'
    )

    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions.xlsx file'
    )

    parser.add_argument(
        '--visual-portions',
        type=str,
        required=True,
        help='Path to folder containing *_visual_portions.json files'
    )

    parser.add_argument(
        '--bins',
        type=int,
        default=5,
        help='Number of bins for visual portion (default: 5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./figures/ndcg_vs_visual_portion.png',
        help='Output path for the figure (default: ./figures/ndcg_vs_visual_portion.png)'
    )

    args = parser.parse_args()

    # Prepare data
    predictions = prepare_data(args.predictions, args.visual_portions)

    # Bin and aggregate
    print(f"\nBinning data into {args.bins} bins...")
    aggregated_data = bin_and_aggregate(predictions, bins=args.bins)
    print("\nAggregated data:")
    print(aggregated_data)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    print("\nCreating plot...")
    plot_ndcg_vs_visual_portion(aggregated_data, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
