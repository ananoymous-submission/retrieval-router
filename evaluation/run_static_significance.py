"""Run the 12 planned static-pipeline significance tests for the paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arms import POLICY_LATENCY_SECONDS
from evaluation.statistical_testing import holm_adjust, paired_comparison

COMPARISONS = [
    (0.1, "10", "ML", "MULTIMODAL-MULTI"),
    (0.1, "10", "MR", "MULTIMODAL_RERANK"),
    (0.1, "10", "TL", "TEXT-MULTI"),
    (0.1, "10", "TR", "TEXT_RERANK"),
    (0.5, "50", "MD", "MULTIMODAL-SINGLE"),
    (0.5, "50", "TD", "TEXT-SINGLE"),
]


def routed_metric(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return np.asarray(
        [row[f"{row['predicted_pipeline']}_{metric}"] for _, row in frame.iterrows()],
        dtype=float,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_analysis(
    predictions_dir: Path,
    output_dir: Path,
    *,
    alpha: float = 0.001,
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    input_paths: dict[str, Path] = {}
    rows = []
    expected_keys = None
    for lam, tag, short, pipeline in COMPARISONS:
        if tag not in frames:
            path = predictions_dir / f"retrievalrouter_l{tag}.xlsx"
            frame = pd.read_excel(path)
            keys = frame["DATASET"].astype(str) + "\x1f" + frame["query"].astype(str)
            if keys.duplicated().any():
                raise ValueError("prediction file contains duplicate (DATASET, query) keys")
            if expected_keys is None:
                expected_keys = set(keys)
            elif set(keys) != expected_keys:
                raise ValueError("prediction files contain different query sets")
            frames[tag] = frame
            input_paths[tag] = path
        frame = frames[tag]
        rr_ndcg = routed_metric(frame, "ndcg")
        rr_latency = routed_metric(frame, "latency") + POLICY_LATENCY_SECONDS
        comparisons = (
            ("ndcg", rr_ndcg, frame[f"{pipeline}_ndcg"].to_numpy(dtype=float)),
            ("latency", frame[f"{pipeline}_latency"].to_numpy(dtype=float), rr_latency),
        )
        for metric, favorable_first, favorable_second in comparisons:
            result = paired_comparison(favorable_first, favorable_second, alpha=alpha)
            rows.append(
                {
                    "lambda": lam,
                    "comparison": f"RR vs {short}",
                    "metric": metric,
                    "rr_mean": float(rr_ndcg.mean() if metric == "ndcg" else rr_latency.mean()),
                    "baseline_mean": float(
                        favorable_second.mean() if metric == "ndcg" else favorable_first.mean()
                    ),
                    "favorable_difference": result["mean_difference"],
                    "normality_p": result["normality"]["p_value"],
                    "normal": result["normality"]["normal"],
                    "test": result["test"],
                    "statistic": result["statistic"],
                    "raw_p": result["p_value"],
                    "effect_size_name": result["effect_size_name"],
                    "effect_size": result["effect_size"],
                }
            )

    results = pd.DataFrame(rows)
    correction = holm_adjust(results["raw_p"].to_numpy(), alpha=alpha)
    results["holm_p"] = correction["adjusted_p"].to_numpy()
    results["significant"] = correction["reject"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "static_results.csv", index=False)
    manifest = {
        "inputs": {str(path): file_sha256(path) for path in input_paths.values()},
        "n_queries": len(next(iter(frames.values()))),
        "alpha": alpha,
        "normality_test": "D'Agostino-Pearson K-squared on paired differences",
        "paired_test_rule": "paired t-test if normal at alpha; otherwise Wilcoxon signed-rank",
        "multiple_comparisons": "Holm across 12 planned tests",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = "# Static-pipeline significance results\n\n" + results.to_markdown(index=False) + "\n"
    (output_dir / "summary.md").write_text(summary)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("evaluation/predictions")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/generated/significance"),
    )
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()
    results = run_analysis(args.predictions_dir, args.output_dir, alpha=args.alpha)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
