"""Single source of truth for pipeline and router-arm names.

`train/`, `evaluation/`, and the notebook all import from here, so the arm set cannot
drift between training and reporting.

Naming is mixed on purpose and must be matched exactly: rerank pipelines use an
underscore (``TEXT_RERANK``), single-vector ones use a hyphen (``TEXT-SINGLE``). Every
pipeline P contributes ``P_ndcg`` / ``P_mrr`` / ``P_precision`` / ``P_recall`` /
``P_latency`` columns to the dataset, and prepare_data writes the matching ``P_reward``.
"""

# The router's action space. The classifier's label i is ARM_NAMES[i], so this order is
# load-bearing: it is baked into every trained checkpoint. BM25 was appended rather than
# inserted precisely so labels 0-3 keep the meaning they had in the pre-BM25 runs.
ARM_NAMES = [
    "MULTIMODAL_RERANK",
    "MULTIMODAL-SINGLE",
    "TEXT_RERANK",
    "TEXT-SINGLE",
    "BM25",
]

# Measured, but never routed to. Reported as reference baselines only (they are the
# accuracy ceiling and the latency worst case).
NON_ARM_PIPELINES = [
    "MULTIMODAL-MULTI",
    "TEXT-MULTI",
]

ALL_PIPELINES = ARM_NAMES + NON_ARM_PIPELINES

# Short labels: heatmap axes (arms only) and point tags in the frontier plot (all pipelines).
SHORT_NAMES = {
    "MULTIMODAL_RERANK": "MR",
    "MULTIMODAL-SINGLE": "MD",
    "TEXT_RERANK": "TR",
    "TEXT-SINGLE": "TD",
    "BM25": "BM",
    "MULTIMODAL-MULTI": "ML",
    "TEXT-MULTI": "TL",
}

# Paper-facing names.
DISPLAY_NAMES = {
    "MULTIMODAL_RERANK": "MM-Rerank",
    "MULTIMODAL-SINGLE": "MM-Dense",
    "MULTIMODAL-MULTI": "MM-Late",
    "TEXT_RERANK": "Text-Rerank",
    "TEXT-SINGLE": "Text-Dense",
    "TEXT-MULTI": "Text-Late",
    "BM25": "BM25",
}

# Shared palette, so a pipeline is the same colour in every figure.
# BM25 uses the publication's distinctive magenta as the lexical outlier.
COLORS = {
    "MULTIMODAL_RERANK": "#C41E3A",
    "MULTIMODAL-SINGLE": "#FF6B35",
    "MULTIMODAL-MULTI": "#8B0000",
    "TEXT_RERANK": "#8B4513",
    "TEXT-SINGLE": "#F4A460",
    "TEXT-MULTI": "#D2691E",
    "BM25": "#D81B60",
}

# Measured query-policy inference overhead, added once to every deployable
# adaptive method after it selects a retrieval arm.
POLICY_LATENCY_SECONDS = 0.015

ARM_SHORT = [SHORT_NAMES[a] for a in ARM_NAMES]
ARM_INDEX = {arm: i for i, arm in enumerate(ARM_NAMES)}


def ndcg_cols(pipelines=ARM_NAMES):
    return [f"{p}_ndcg" for p in pipelines]


def latency_cols(pipelines=ARM_NAMES):
    return [f"{p}_latency" for p in pipelines]


def reward_cols(pipelines=ARM_NAMES):
    return [f"{p}_reward" for p in pipelines]


NDCG_COLS = ndcg_cols()
LATENCY_COLS = latency_cols()
REWARD_COLS = reward_cols()
