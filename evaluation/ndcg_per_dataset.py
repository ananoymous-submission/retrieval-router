import pandas as pd
import numpy as np
import sys

def main():
    # Default to lambda=0.0, can pass as argument
    lambda_val = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    filepath = f'./evaluation/predictions/predictions_rerank.xlsx'

    print(f"Loading {filepath}...")
    df = pd.read_excel(filepath)

    # Column mapping for display
    pipeline_names = {
        'TEXT-SINGLE': 'Text-Dense',
        'TEXT-MULTI': 'Text-Late',
        'MULTIMODAL-SINGLE': 'MM-Dense',
        'MULTIMODAL-MULTI': 'MM-Late',
        "TEXT_RERANK": "Text-Rerank",
        "MULTIMODAL_RERANK": "MM-Rerank"
    }

    # Get RAG-Mixer predicted NDCG per row
    df['RAG-Mixer'] = df.apply(
        lambda row: row[f"{row['predicted_strategy']}_ndcg"], axis=1
    )

    # Group by dataset
    results = []
    for dataset in df['DATASET'].unique():
        subset = df[df['DATASET'] == dataset]
        row = {'Dataset': dataset}

        for pipeline, display_name in pipeline_names.items():
            row[display_name] = subset[f'{pipeline}_ndcg'].mean()

        row['RAG-Mixer'] = subset['RAG-Mixer'].mean()
        results.append(row)

    result_df = pd.DataFrame(results)

    # Order columns
    col_order = ['Dataset', 'Text-Dense', 'Text-Late', 'MM-Dense', 'MM-Late', "Text-Rerank", "MM-Rerank", 'RAG-Mixer']
    result_df = result_df[col_order]

    # Sort by dataset name
    result_df = result_df.sort_values('Dataset')

    # Format for display
    print(f"\nNDCG per Dataset (Lambda={lambda_val/100:.2f}):\n")
    print(result_df.to_string(index=False, float_format='%.3f'))

    # Save
    output_file = f'./evaluation/ndcg_per_dataset_lambda_{lambda_val:02d}.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
