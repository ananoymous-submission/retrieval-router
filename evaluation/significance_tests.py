from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, levene, ks_2samp
from statsmodels.stats.anova import AnovaRM
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_best_model_per_metric(ratings_df: pd.DataFrame) -> Dict[str, str]:
    best_models = {}
    for metric in ratings_df.columns:
        mean_scores = ratings_df[metric].apply(lambda x: np.mean(np.array(x)))
        best_models[metric] = mean_scores.idxmax()
    return best_models


def prepare_long_format_data(ratings_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    long_data_list = []
    n_subjects = len(next(iter(ratings_df[metric])))
    for subject_idx in range(n_subjects):
        for model_name in ratings_df.index:
            value = ratings_df.loc[model_name, metric][subject_idx]
            long_data_list.append({
                'Subject': subject_idx,
                'Model': model_name,
                'Value': value
            })
    return pd.DataFrame(long_data_list)


def test_normality(data: np.ndarray) -> bool:
    _, p_value = ks_2samp(data, np.random.normal(np.mean(data), np.std(data), len(data)))
    return p_value >= 0.05


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Args:
        group1: First group's data
        group2: Second group's data
        
    Returns:
        Cohen's d effect size value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return d


def perform_group_significance_test(ratings_df: pd.DataFrame, metric: str) -> Tuple[str, float]:
    methods = list(ratings_df.index)

    # If only two methods, use paired test
    if len(methods) < 3:
        arr1 = np.array(ratings_df.loc[methods[0], metric])
        arr2 = np.array(ratings_df.loc[methods[1], metric])
        norm1 = test_normality(arr1)
        norm2 = test_normality(arr2)
        if norm1 and norm2:
            _, lev_p = levene(arr1, arr2)
            if lev_p >= 0.05:
                _, p_value = ttest_rel(arr1, arr2)
                return "Paired t-test", p_value
        _, p_value = wilcoxon(arr1, arr2)
        return "Wilcoxon Signed-Rank Test", p_value

    # For three or more, test normality
    normal_tests = []
    for method in methods:
        data = np.array(ratings_df.loc[method, metric])
        is_normal = test_normality(data)
        normal_tests.append(is_normal)

    # If all normal, try RM-ANOVA
    if all(normal_tests):
        long_data = prepare_long_format_data(ratings_df, metric)
        try:
            anova = AnovaRM(long_data, 'Value', 'Subject', within=['Model']).fit()
            return "Repeated Measures ANOVA", anova.anova_table['Pr > F'][0]
        except Exception as e:
            logger.warning(f"RM ANOVA failed: {e}, falling back to Friedman test")
            
    # Friedman fallback
    friedman_data = [np.array(ratings_df.loc[m, metric]) for m in methods]
    stat, p_value = friedmanchisquare(*friedman_data)
    return "Friedman Test", p_value


def perform_pairwise_comparison(
    best_scores: List[float],
    comparison_scores: List[float]
) -> Tuple[str, float, float]:
    best_array = np.array(best_scores)
    comp_array = np.array(comparison_scores)
    
    # Calculate effect size
    effect_size = calculate_cohens_d(best_array, comp_array)
    
    norm1 = test_normality(best_array)
    norm2 = test_normality(comp_array)
    if norm1 and norm2:
        _, lev_p = levene(best_array, comp_array)
        if lev_p >= 0.05:
            _, p_value = ttest_rel(best_array, comp_array)
            return "Paired t-test", p_value, effect_size
    _, p_value = wilcoxon(best_array, comp_array)
    return "Wilcoxon Signed-Rank Test", p_value, effect_size


def interpret_effect_size(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        effect_size: The calculated Cohen's d value
        
    Returns:
        String interpretation of the effect size
    """
    if abs(effect_size) < 0.2:
        return "negligible"
    elif abs(effect_size) < 0.5:
        return "small"
    elif abs(effect_size) < 0.8:
        return "medium"
    else:
        return "large"


def compare_models(ratings_df: pd.DataFrame) -> pd.DataFrame:
    comparisons_data: List[Dict] = []
    best_models = get_best_model_per_metric(ratings_df)
    for metric in ratings_df.columns:
        best_model = best_models[metric]
        group_test, group_p = perform_group_significance_test(ratings_df, metric)
        if group_p < 0.05 and len(ratings_df.index) > 1:
            for model in ratings_df.index:
                if model == best_model:
                    continue
                try:
                    test_used, p_val, effect_size = perform_pairwise_comparison(
                        ratings_df.loc[best_model, metric],
                        ratings_df.loc[model, metric]
                    )
                    effect_interpretation = interpret_effect_size(effect_size)
                    
                    comparisons_data.append({
                        "Metric": metric,
                        "Group Test": group_test,
                        "Group p": group_p,
                        "Best Method": best_model,
                        "Compared Method": model,
                        "Pair Test": test_used,
                        "Pair p": p_val,
                        "Significant": p_val < 0.05,
                        "Effect Size (Cohen's d)": effect_size,
                        "Effect Interpretation": effect_interpretation
                    })
                except Exception as e:
                    logger.warning(f"Comparison {best_model} vs {model} failed: {e}")
        else:
            comparisons_data.append({
                "Metric": metric,
                "Group Test": group_test,
                "Group p": group_p,
                "Best Method": best_model,
                "Compared Method": None,
                "Pair Test": None,
                "Pair p": None,
                "Significant": False,
                "Effect Size (Cohen's d)": None,
                "Effect Interpretation": None
            })
    return pd.DataFrame(comparisons_data)
