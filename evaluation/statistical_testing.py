"""Deterministic repeated-measures statistical testing utilities."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import normaltest, rankdata, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


def align_method_scores(methods: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align method scores by query key and reject non-identical subject sets."""
    if not methods:
        raise ValueError("at least one method is required")

    first_name, first = next(iter(methods.items()))
    if not first.index.is_unique:
        raise ValueError(f"{first_name} contains duplicate query keys")
    expected_keys = set(first.index)

    aligned = {}
    for name, scores in methods.items():
        if not scores.index.is_unique:
            raise ValueError(f"{name} contains duplicate query keys")
        if set(scores.index) != expected_keys:
            raise ValueError("all methods must contain identical query keys")
        aligned[name] = scores.reindex(first.index).to_numpy()

    return pd.DataFrame(aligned, index=first.index)


def normality_diagnostic(values, alpha: float = 0.01) -> dict:
    """Return a deterministic normality diagnostic for a one-dimensional sample."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 8:
        raise ValueError("normality diagnostic requires at least 8 finite values")
    statistic, p_value = normaltest(array)
    return {
        "test": "D'Agostino-Pearson K-squared",
        "n": int(len(array)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "normal": bool(p_value >= alpha),
    }


def paired_comparison(first, second, alpha: float = 0.01) -> dict:
    """Choose a two-sided paired test from the distribution of paired differences."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("paired samples must have equal shape")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("paired samples must be finite")

    differences = first_array - second_array
    diagnostic = normality_diagnostic(differences, alpha=alpha)
    mean_difference = float(differences.mean())

    if diagnostic["normal"]:
        statistic, p_value = ttest_rel(first_array, second_array)
        sd = differences.std(ddof=1)
        effect_size = float(mean_difference / sd) if sd else 0.0
        test_name = "paired_t"
        effect_name = "cohen_dz"
    else:
        statistic, p_value = wilcoxon(
            first_array, second_array, alternative="two-sided", zero_method="pratt"
        )
        nonzero = differences != 0
        ranks = rankdata(np.abs(differences[nonzero]))
        positive = ranks[differences[nonzero] > 0].sum()
        negative = ranks[differences[nonzero] < 0].sum()
        rank_total = positive + negative
        effect_size = float((positive - negative) / rank_total) if rank_total else 0.0
        test_name = "wilcoxon"
        effect_name = "rank_biserial"

    return {
        "test": test_name,
        "n": int(len(differences)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "mean_difference": mean_difference,
        "effect_size_name": effect_name,
        "effect_size": effect_size,
        "normality": diagnostic,
    }


def holm_adjust(p_values, alpha: float = 0.01) -> pd.DataFrame:
    """Apply Holm familywise-error correction while preserving input order."""
    raw = np.asarray(p_values, dtype=float)
    reject, adjusted, _, _ = multipletests(raw, alpha=alpha, method="holm")
    return pd.DataFrame({"raw_p": raw, "adjusted_p": adjusted, "reject": reject})



