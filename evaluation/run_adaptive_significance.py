"""Test RR against the adaptive baseline on nDCG and latency at each lambda."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arms import ARM_NAMES, POLICY_LATENCY_SECONDS
from evaluation.statistical_testing import holm_adjust, paired_comparison

LAMBDAS = [(0.0, "00"), (0.1, "10"), (0.3, "30"), (0.5, "50"), (0.7, "70")]


def query_keys(frame: pd.DataFrame) -> pd.Index:
    keys = frame["DATASET"].astype(str) + "\x1f" + frame["query"].astype(str)
    if keys.duplicated().any():
        raise ValueError("duplicate (DATASET, query) keys")
    return pd.Index(keys)


def baseline_selections(frame: pd.DataFrame) -> dict[float, np.ndarray]:
    probabilities = frame[[f"pred_prob_{arm}" for arm in ARM_NAMES]].to_numpy()
    cheap = ARM_NAMES.index("BM25")
    expensive = [index for index in range(len(ARM_NAMES)) if index != cheap]
    best_expensive = np.asarray(expensive)[np.argmax(probabilities[:, expensive], axis=1)]
    selections = []
    budgets = []
    for tau in np.linspace(0.0, 1.0, 4001):
        selected = np.where(probabilities[:, cheap] >= tau, cheap, best_expensive)
        selections.append(selected)
        budgets.append((selected != cheap).mean())
    budgets = np.asarray(budgets)
    return {
        lam: selections[int(np.abs(budgets - (1.0 - lam)).argmin())]
        for lam, _ in LAMBDAS
    }


def run(predictions_dir: Path, output: Path, *, alpha: float = 0.001) -> pd.DataFrame:
    baseline = pd.read_excel(predictions_dir / "baseline.xlsx")
    baseline.index = query_keys(baseline)
    selected_baseline = baseline_selections(baseline)
    rows = []
    for lam, tag in LAMBDAS:
        router = pd.read_excel(predictions_dir / f"retrievalrouter_l{tag}.xlsx")
        router.index = query_keys(router)
        router = router.reindex(baseline.index)
        router_indices = np.asarray(
            [ARM_NAMES.index(name) for name in router["predicted_pipeline"]]
        )
        query_rows = np.arange(len(router))
        for metric in ("ndcg", "latency"):
            matrix = router[[f"{arm}_{metric}" for arm in ARM_NAMES]].to_numpy()
            router_scores = matrix[query_rows, router_indices]
            baseline_scores = matrix[query_rows, selected_baseline[lam]]
            if metric == "latency":
                router_scores = router_scores + POLICY_LATENCY_SECONDS
                baseline_scores = baseline_scores + POLICY_LATENCY_SECONDS
            favorable_first, favorable_second = (
                (router_scores, baseline_scores)
                if metric == "ndcg"
                else (baseline_scores, router_scores)
            )
            result = paired_comparison(favorable_first, favorable_second, alpha=alpha)
            rows.append(
                {
                    "lambda": lam,
                    "metric": metric,
                    "rr_mean": router_scores.mean(),
                    "baseline_mean": baseline_scores.mean(),
                    "favorable_difference": result["mean_difference"],
                    "normality_p": result["normality"]["p_value"],
                    "normal": result["normality"]["normal"],
                    "test": result["test"],
                    "raw_p": result["p_value"],
                    "effect_size_name": result["effect_size_name"],
                    "effect_size": result["effect_size"],
                }
            )
    results = pd.DataFrame(rows)
    correction = holm_adjust(results["raw_p"].to_numpy(), alpha=alpha)
    results["holm_p"] = correction["adjusted_p"].to_numpy()
    results["significant"] = correction["reject"].to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("evaluation/predictions")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/generated/significance/baseline_results.csv"),
    )
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()
    print(run(args.predictions_dir, args.output, alpha=args.alpha).to_string(index=False))


if __name__ == "__main__":
    main()
