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


def determine_selected_pipeline(row, mode='rag-mixer'):
    """
    Determine which pipeline was selected for this query.

    Args:
        row: DataFrame row with predicted_strategy and true_strategy columns
        mode: 'rag-mixer' to use predicted selections, 'oracle' for optimal selections

    Returns:
        str: Name of selected pipeline
    """
    if mode == 'oracle':
        return row['true_strategy']
    else:
        return row['predicted_strategy']


def prepare_data(predictions_path, visual_portions_folder, mode='rag-mixer'):
    """
    Prepare data for plotting by combining predictions with visual portions.

    Args:
        predictions_path: Path to predictions Excel file
        visual_portions_folder: Path to folder with visual portion JSONs
        mode: 'rag-mixer' or 'oracle'

    Returns:
        pd.DataFrame: DataFrame with query, visual_portion, and selected_pipeline
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

    # Determine which pipeline was selected
    mode_label = 'Oracle' if mode == 'oracle' else 'RAG-MIXER'
    print(f"\nDetermining selected pipelines ({mode_label})...")
    predictions['selected_pipeline'] = predictions.apply(
        lambda row: determine_selected_pipeline(row, mode), axis=1
    )

    return predictions


def bin_and_aggregate_selections(predictions, bins=10):
    """
    Bin queries by visual portion and count pipeline selections in each bin.

    Args:
        predictions: DataFrame with visual_portion and selected_pipeline columns
        bins: Number of bins for visual portion

    Returns:
        pd.DataFrame: Aggregated data with bin centers and counts for each pipeline
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

    # Count pipeline selections for each bin
    pipelines = ['TEXT-RERANK', 'MULTIMODAL-RERANK']

    aggregated_data = []
    for bin_idx in range(bins):
        bin_data = predictions[predictions['visual_bin'] == bin_idx]

        if len(bin_data) == 0:
            continue

        row = {'bin_center': bin_centers[bin_idx], 'total_count': len(bin_data)}

        # Count how many queries selected each pipeline
        for pipeline in pipelines:
            count = len(bin_data[bin_data['selected_pipeline'] == pipeline])
            row[pipeline] = count

        aggregated_data.append(row)

    return pd.DataFrame(aggregated_data)


def plot_pipeline_selection(aggregated_data, output_path=None, mode='rag-mixer'):
    """
    Create stacked bar chart of pipeline selection vs visual portion.

    Args:
        aggregated_data: DataFrame with bin_center and counts for each pipeline
        output_path: Optional path to save the figure
        mode: 'rag-mixer' or 'oracle' for title
    """
    pipelines = ['TEXT-RERANK', 'MULTIMODAL-RERANK']
    colors = ['#3498db', '#e74c3c']  # Blue for Text, Red for Multimodal

    # Rename for legend
    label_map = {
        'TEXT-RERANK': 'Text Rerank',
        'MULTIMODAL-RERANK': 'Multimodal Rerank'
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate bin width for bar chart
    bin_width = 0.08  # Approximate width for 10 bins from 0 to 1

    # Create stacked bars
    bottom = np.zeros(len(aggregated_data))

    for pipeline, color in zip(pipelines, colors):
        values = aggregated_data[pipeline].values
        label = label_map.get(pipeline, pipeline)

        ax.bar(
            aggregated_data['bin_center'],
            values,
            width=bin_width,
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.9,
            edgecolor='white',
            linewidth=0.5
        )

        bottom += values

    # Styling
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlabel('Portion of Visual Content', fontsize=24, fontweight='bold')
    ax.set_ylabel('Number of Queries', fontsize=24, fontweight='bold')
    title_label = 'Oracle' if mode == 'oracle' else 'RAG-MIXER'
    ax.legend(loc='best', fontsize=24, frameon=False, edgecolor='black')
    ax.set_xlim(-0.05, 1.05)
    # Add 10% margin above max bar
    max_height = aggregated_data['total_count'].max()
    ax.set_ylim(0, max_height * 1.1)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot pipeline selection by RAG-MIXER across visual portion levels'
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
        default=10,
        help='Number of bins for visual portion (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./figures/pipeline_selection.png',
        help='Output path for the figure (default: ./figures/pipeline_selection.png)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['rag-mixer', 'oracle'],
        default='rag-mixer',
        help='Mode: rag-mixer (predicted selections) or oracle (optimal selections)'
    )

    args = parser.parse_args()

    # Prepare data
    predictions = prepare_data(args.predictions, args.visual_portions, args.mode)

    # Bin and aggregate
    print(f"\nBinning data into {args.bins} bins...")
    aggregated_data = bin_and_aggregate_selections(predictions, bins=args.bins)
    print("\nAggregated data:")
    print(aggregated_data)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    print("\nCreating plot...")
    plot_pipeline_selection(aggregated_data, args.output, args.mode)

    print("\nDone!")


if __name__ == "__main__":
    main()
