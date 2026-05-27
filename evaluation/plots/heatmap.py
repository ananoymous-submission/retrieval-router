## NDCG Selection vs Non-Selection Heatmap Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_selection_performance_heatmap(router, base_pipelines, metric="ndcg"):
    """
    Create heatmap showing raw performance of all pipelines for different router selection scenarios.
    Columns: Router selection scenarios (which pipeline the router selected)
    Rows: All pipelines (showing their performance in each scenario)

    Args:
        router: DataFrame with router predictions
        base_pipelines: DataFrame with pipeline performance metrics
        metric: Either "ndcg" or "latency" (default: "ndcg")
    """
    pipeline_names = ["TEXT_RERANK", "MULTIMODAL_RERANK"]

    # Create display names with abbreviations
    display_names = ["TR", "MR"]

    # Define selection scenarios based on router predictions
    selection_scenarios = {
        'TR': router["predicted_label"] == 0,
        'MR': router["predicted_label"] == 1
    }

    # Create scenario names with query counts
    scenario_names_with_counts = []
    for name, mask in selection_scenarios.items():
        count = mask.sum()
        scenario_names_with_counts.append(f"{name} ({count})")

    # Update the dictionary with new names
    selection_scenarios_with_counts = dict(zip(scenario_names_with_counts, selection_scenarios.values()))

    # Create data matrix: pipelines (rows) x selection scenarios (columns)
    heatmap_data = np.zeros((len(pipeline_names), len(selection_scenarios)))

    for col_idx, (scenario_name, scenario_mask) in enumerate(selection_scenarios_with_counts.items()):
        # Get queries for this scenario
        scenario_queries = router[scenario_mask].index

        if len(scenario_queries) > 0:
            for row_idx, pipeline in enumerate(pipeline_names):
                metric_col = f"{pipeline}_{metric}"

                if metric_col in base_pipelines.columns:
                    pipeline_metric = base_pipelines.loc[scenario_queries, metric_col].dropna()

                    if len(pipeline_metric) > 0:
                        heatmap_data[row_idx, col_idx] = pipeline_metric.mean()

    # Transpose the data: selections (rows) x pipelines (columns)
    heatmap_data_transposed = heatmap_data.T

    return heatmap_data_transposed, scenario_names_with_counts, display_names

def plot_selection_heatmap(router, base_pipelines, metric="ndcg"):
    """
    Main function to create and display the router selection correlation heatmap

    Args:
        router: DataFrame with router predictions
        base_pipelines: DataFrame with pipeline performance metrics
        metric: Either "ndcg" or "latency" (default: "ndcg")
    """
    # Create the analysis
    heatmap_data, scenario_names, display_names = create_selection_performance_heatmap(router, base_pipelines, metric)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))

    # Determine colorbar label based on metric
    metric_label = metric.upper()

    # Create heatmap showing pipeline performance across router selection scenarios
    ax = sns.heatmap(
        heatmap_data,
        xticklabels=display_names,
        yticklabels=scenario_names,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric_label},
        annot_kws={'size': 24, 'weight': 'bold'}
    )

    # Make axis labels bold
    ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, fontweight='bold')
    # Make colorbar label bold
    cbar = ax.collections[0].colorbar
    cbar.set_label(metric_label, fontsize=24, fontweight='bold')
    cbar.ax.yaxis.label.set_fontsize(24)
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.tick_params(labelsize=24)
    
    plt.xticks(rotation=0, ha='center', fontsize=24, fontweight='bold')
    plt.yticks(rotation=0, fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def create_reward_heatmap(router, base_pipelines, lambda_val):
    """
    Create heatmap showing reward values for all pipelines across router selection scenarios.
    Reward = (1 - lambda) * ndcg - lambda * normalized_latency

    Latency normalization: per-row sum-based (matching train/prepare_data.py)

    Args:
        router: DataFrame with router predictions (must have 'predicted_label' column)
        base_pipelines: DataFrame with pipeline performance metrics
        lambda_val: Lambda value for reward calculation (0 to 1)

    Returns:
        heatmap_data_transposed: Matrix of reward values (rows=selections, cols=pipelines)
        scenario_names_with_counts: List of scenario names with query counts
        display_names: List of pipeline display names
    """
    pipeline_names = ["TEXT_RERANK", "MULTIMODAL_RERANK"]
    display_names = ["TR", "MR"]

    # Define selection scenarios based on router predictions
    selection_scenarios = {
        'TR': router["predicted_label"] == 0,
        'MR': router["predicted_label"] == 1
    }

    # Create scenario names with query counts
    scenario_names_with_counts = []
    for name, mask in selection_scenarios.items():
        count = mask.sum()
        scenario_names_with_counts.append(f"{name} ({count})")

    selection_scenarios_with_counts = dict(zip(scenario_names_with_counts, selection_scenarios.values()))

    # Get latency columns for per-row normalization
    latency_cols = [f"{p}_latency" for p in pipeline_names]
    ndcg_cols = [f"{p}_ndcg" for p in pipeline_names]

    # Create data matrix: pipelines (rows) x selection scenarios (columns)
    heatmap_data = np.zeros((len(pipeline_names), len(selection_scenarios)))

    for col_idx, (scenario_name, scenario_mask) in enumerate(selection_scenarios_with_counts.items()):
        scenario_queries = router[scenario_mask].index
        scenario_queries = scenario_queries.intersection(base_pipelines.index)

        if len(scenario_queries) > 0:
            # Get all latencies and NDCGs for these queries
            latencies = base_pipelines.loc[scenario_queries, latency_cols].values
            ndcgs = base_pipelines.loc[scenario_queries, ndcg_cols].values

            # Per-row normalize latency (sum-based, matching prepare_data.py)
            latency_sum = latencies.sum(axis=1, keepdims=True)
            normalized_latencies = latencies / latency_sum

            # Calculate rewards for all pipelines
            rewards = (1 - lambda_val) * ndcgs - lambda_val * normalized_latencies

            # Store mean reward for each pipeline
            for row_idx in range(len(pipeline_names)):
                heatmap_data[row_idx, col_idx] = rewards[:, row_idx].mean()

    # Transpose: selections (rows) x pipelines (columns)
    heatmap_data_transposed = heatmap_data.T

    return heatmap_data_transposed, scenario_names_with_counts, display_names


def plot_reward_heatmap(router, base_pipelines, lambda_val):
    """
    Create and display the reward-based heatmap for router selection analysis.

    Args:
        router: DataFrame with router predictions
        base_pipelines: DataFrame with pipeline performance metrics
        lambda_val: Lambda value for reward calculation
    """
    heatmap_data, scenario_names, display_names = create_reward_heatmap(
        router, base_pipelines, lambda_val
    )

    plt.figure(figsize=(12, 8))

    ax = sns.heatmap(
        heatmap_data,
        xticklabels=display_names,
        yticklabels=scenario_names,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': f'nDCG@5'},
        annot_kws={'size': 24, 'weight': 'bold'}
    )

    ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, fontweight='bold')

    cbar = ax.collections[0].colorbar
    cbar.set_label(f'nDCG@5', fontsize=24, fontweight='bold')
    cbar.ax.yaxis.label.set_fontsize(24)
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.tick_params(labelsize=24)

    plt.xticks(rotation=0, ha='center', fontsize=24, fontweight='bold')
    plt.yticks(rotation=0, fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.show()
