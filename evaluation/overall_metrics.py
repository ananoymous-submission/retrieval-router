import os
import pandas as pd
import numpy as np


def calculate_metrics(df, col_prefix):
    """
    Calculate mean metrics for a specific pipeline prefix (e.g. 'TEXT-SINGLE')
    """
    return {
        'NDCG': df[f'{col_prefix}_ndcg'].mean(),
        'MRR': df[f'{col_prefix}_mrr'].mean(),
        'Recall': df[f'{col_prefix}_recall'].mean(),
        'Latency': df[f'{col_prefix}_latency'].mean(),
        'Latency_P95': df[f'{col_prefix}_latency'].quantile(0.95),
        'Latency_P99': df[f'{col_prefix}_latency'].quantile(0.99),
    }


def main():
    predictions_file = '/Users/emrekuru/Developer/Kanzy/Research/RAG-Mixer/evaluation/predictions/predictions_rerank.xlsx'
    output_file = './evaluation/metrics_summary.xlsx'
    
    if not os.path.exists(predictions_file):
        print(f"Prediction file not found: {predictions_file}")
        return

    base_pipelines = ['TEXT-SINGLE', 'TEXT-MULTI', 'MULTIMODAL-SINGLE', 'MULTIMODAL-MULTI', 'TEXT_RERANK', 'MULTIMODAL_RERANK']
    
    all_results = []
    
    print(f"Loading {predictions_file}...")
    df = pd.read_excel(predictions_file)

    # RAG-Mixer metrics (based on predicted_strategy)
    if 'predicted_strategy' in df.columns:
        rag_metrics = {'NDCG': [], 'MRR': [], 'Recall': [], 'Latency': []}

        for idx, row in df.iterrows():
            strat = row['predicted_strategy']
            rag_metrics['NDCG'].append(row[f"{strat}_ndcg"])
            rag_metrics['MRR'].append(row[f"{strat}_mrr"])
            rag_metrics['Recall'].append(row[f"{strat}_recall"])
            rag_metrics['Latency'].append(row[f"{strat}_latency"])

        rag_series_lat = pd.Series(rag_metrics['Latency'])
        all_results.append({
            'System': 'RAG-Mixer',
            'Type': 'Selection',
            'NDCG': np.mean(rag_metrics['NDCG']),
            'MRR': np.mean(rag_metrics['MRR']),
            'Recall': np.mean(rag_metrics['Recall']),
            'Latency': rag_series_lat.mean(),
            'Latency_P95': rag_series_lat.quantile(0.95),
            'Latency_P99': rag_series_lat.quantile(0.99)
        })
    else:
        print("Warning: No predicted_strategy column found")

    # Base pipeline metrics
    for pipeline in base_pipelines:
        if f"{pipeline}_ndcg" in df.columns:
            metrics = calculate_metrics(df, pipeline)
            all_results.append({
                'System': pipeline,
                'Type': 'Base Pipeline',
                'NDCG': metrics['NDCG'],
                'MRR': metrics['MRR'],
                'Recall': metrics['Recall'],
                'Latency': metrics['Latency'],
                'Latency_P95': metrics['Latency_P95'],
                'Latency_P99': metrics['Latency_P99']
            })
        else:
            print(f"Warning: Pipeline {pipeline} not found in data")

    # Save
    res_df = pd.DataFrame(all_results)
    
    col_order = ['System', 'Type', 'NDCG', 'MRR', 'Recall', 'Latency', 'Latency_P95', 'Latency_P99']
    res_df = res_df[col_order]
    
    res_df.to_excel(output_file, index=False)
    print(f"\nSaved metrics summary to {output_file}")
    
    print("\nResults:")
    print(res_df.to_string())


if __name__ == "__main__":
    main()
