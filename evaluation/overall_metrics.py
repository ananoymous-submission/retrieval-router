"""Overall routed-vs-baseline metrics for a predictions file.

Usage:
    python -m evaluation.overall_metrics
    python -m evaluation.overall_metrics --predictions evaluation/predictions/retrievalrouter_l30.xlsx
"""
import os
import argparse

import pandas as pd
import numpy as np

from arms import ALL_PIPELINES, DISPLAY_NAMES, POLICY_LATENCY_SECONDS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PREDICTIONS = os.path.join(REPO_ROOT, "evaluation/predictions/retrievalrouter_l10.xlsx")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation/metrics_summary.xlsx")

METRICS = ["ndcg", "mrr", "recall", "latency"]
COL_ORDER = ["System", "Type", "NDCG", "MRR", "Recall", "Latency", "Latency_P95", "Latency_P99"]


def _summarize(ndcg, mrr, recall, latency):
    latency = pd.Series(latency)
    return {
        "NDCG": np.mean(ndcg),
        "MRR": np.mean(mrr),
        "Recall": np.mean(recall),
        "Latency": latency.mean(),
        "Latency_P95": latency.quantile(0.95),
        "Latency_P99": latency.quantile(0.99),
    }


def calculate_metrics(df, pipeline):
    """Mean metrics for one fixed pipeline (e.g. 'TEXT-SINGLE')."""
    return _summarize(*(df[f"{pipeline}_{m}"] for m in METRICS))


def routed_metrics(df):
    """What the router actually achieved: on each query, score the arm it picked."""
    if "predicted_pipeline" not in df.columns:
        raise KeyError("predictions file has no 'predicted_pipeline' column")

    picked = df["predicted_pipeline"]
    unknown = set(picked.unique()) - set(ALL_PIPELINES)
    if unknown:
        raise ValueError(f"predicted_pipeline contains unknown pipelines: {sorted(unknown)}")

    # Vectorised gather: for each row, take column f"{picked_arm}_{metric}".
    rows = np.arange(len(df))
    cols = picked.map({p: i for i, p in enumerate(ALL_PIPELINES)}).to_numpy()
    gathered = {
        m: df[[f"{p}_{m}" for p in ALL_PIPELINES]].to_numpy()[rows, cols]
        for m in METRICS
    }
    gathered["latency"] = gathered["latency"] + POLICY_LATENCY_SECONDS
    return _summarize(*(gathered[m] for m in METRICS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Loading {args.predictions}...")
    df = pd.read_excel(args.predictions)

    missing = [p for p in ALL_PIPELINES if f"{p}_ndcg" not in df.columns]
    if missing:
        raise KeyError(f"predictions file is missing columns for pipelines: {missing}")

    rows = [{"System": "RetrievalRouter", "Type": "Selection", **routed_metrics(df)}]
    for pipeline in ALL_PIPELINES:
        rows.append({
            "System": DISPLAY_NAMES[pipeline],
            "Type": "Base Pipeline",
            **calculate_metrics(df, pipeline),
        })

    res_df = pd.DataFrame(rows)[COL_ORDER]
    res_df.to_excel(args.output, index=False)

    print(f"\nSaved metrics summary to {args.output}\n")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
