"""Arabzadeh et al., CIKM 2021 -- the adaptive baseline (NOT ours).

    Negar Arabzadeh, Xinyi Yan, Charles L. A. Clarke. "Predicting Efficiency/Effectiveness
    Trade-offs for Dense vs. Sparse Retrieval Strategy Selection." CIKM 2021. arXiv:2109.10739.
    Code: https://github.com/Narabzad/Retrieval-Strategy-Selection

The closest prior query-only adaptive retrieval baseline to RetrievalRouter.
Verified against their source, not a secondhand summary.

THEIR LABEL, from their README verbatim:

    we set the query class=0 (sparse retriever) if the sparse retriever can retrieve and rank
    any relevant document among top-T retrieved documents. Otherwise, we prefer the more
    expensive and complex dense retriever with class label=1.

with T=50. In one line: take the cheap retriever if it works, otherwise escalate.

THEIR MODEL is not really a cross-encoder. `train_sparse_vs_dense.py` feeds
`InputExample(texts=[qtext])` -- a *single* text -- to `CrossEncoder('bert-base-uncased',
num_labels=2)`. sentence-transformers' CrossEncoder just wraps AutoModelForSequenceClassification;
with one text there is nothing to cross-attend, so it collapses to plain BERT + a head on
[CLS]. It only genuinely cross-encodes in their *hybrid* script (`texts=[qtext, doctext]`),
which is a post-retrieval decision and therefore cannot be a pre-retrieval router. Their method
reduces to a query-text classifier over strategies -- structurally the same architecture used
by RetrievalRouter,
which is exactly why we can run it on our pipeline and isolate the training signal.

VERIFIED AGAINST THE PAPER (arXiv:2109.10739), not just the repo. Their own words:

  LABEL   "Let the first relevant retrieved passage within S_q^K be F_q. If F_q appears above
           a threshold T in S_q^K, we label q as 'Sparse Retriever'. If F_q appears below the
           threshold, OR IF S_q^K DOES NOT CONTAIN A RELEVANT PASSAGE, we label q [Dense]."
  BUDGET  "The classifiers produce probabilities... we pick a threshold between 0.0 and 1.0.
           As the threshold is varied from 0.0 to 1.0, the dense retrieval strategy is selected
           for a larger and larger fraction of the queries... We can view this fraction as a
           'budget'... For a budget of 0, the sparse retriever is always used. For a budget of
           1, the alternative retriever is always used."
  CONTROL "As a baseline, we randomly assign a retriever at the rate given by the budget."
  METRIC  recall@1000 (first-stage pooling; a reranker runs later).

So: the label, the probability-threshold budget sweep (sweep_tau below), and the
random-allocation control (random_allocation below) are all implemented as specified.

HOW WE ADAPTED IT (every deviation, stated):
  * 2 arms -> our 5. "Cheap" becomes "cheapest by measured mean latency", derived from data
    rather than hardcoded.
  * T=50 on a 1000-deep list -> RANK_THRESHOLD on our 5-deep qrels. Their T is a free
    parameter; ours is exposed as RANK_THRESHOLD rather than silently fixed. See the note below.
  * All-fail queries: their rule escalates to the EXPENSIVE arm. We label the CHEAPEST instead.
    Those queries score 0 whichever arm runs them, so escalating buys zero accuracy at maximum
    cost and is strictly dominated. Measured: identical nDCG (0.7745), 19% lower latency
    (0.2554s -> 0.2072s). This makes the baseline stronger than the literal rule, not weaker.
  * The same two-epoch training budget as RetrievalRouter.

NOTE ON RANK_THRESHOLD. Their T controls how demanding "success" is, and it moves the label's
ceiling a lot: T=5 (gold anywhere in the top-5) yields a ceiling of nDCG 0.777, while T=1 (gold
at rank 1) yields 0.852. It is a real knob, so it is a named constant here, not a magic number.

NOTE ON LAMBDA. There isn't one. "Cheapest that works" carries no accuracy/latency weight, so
this objective yields a single model and a single operating point -- unlike RetrievalRouter,
frontier comes from lambda. To place it on the same plot, use the inference-time cost knob
`sweep_tau` at the bottom of this file.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from arms import ARM_NAMES, NDCG_COLS, LATENCY_COLS
from train.objectives import Objective

BASELINE_LABEL_COL = "baseline_label"

#: Rank of the first relevant doc at or above which the cheap arm counts as "successful".
#:
#: Their T=50 against a 1000-deep list (T/K = 5%) -- deliberately far STRICTER than their
#: metric's cutoff, because T=K would label nearly every query "sparse" and the label would
#: carry no information. Our qrels are only 5 deep (nDCG@5), so T can be 1..5 and no
#: 5%-equivalent exists. T=5 would be the degenerate T=K case (the cheap arm "succeeds" merely
#: by retrieving the doc at all, even at rank 5 where nDCG is ~0.39). T=2 keeps the label
#: meaningfully strict -- the cheap arm must rank a relevant doc 1st or 2nd.
#:
#: VERIFIED AGAINST THE PAPER (not just the repo):
#:   "For the results in this paper we use a threshold of T = 50, but similar results are
#:    obtained for other values (100, 150, and 200)."
#: Their retrieval depth is K = 1000 (recall@1000), so their tested band is T/K in [5%, 20%]
#: -- and they only ever loosen T beyond 50, never tighten it.
#:
#: Our K = 5 (nDCG@5), so T=1 -> 20% sits exactly at the top of their band, while T=2 -> 40%
#: is LOOSER than anything they tested. T=1 is therefore the faithful default. T=2 is kept as
#: a sensitivity run, and the two agree closely (swept-curve peaks 0.7156 vs 0.7080), which
#: independently replicates their own "similar results for other values" robustness claim.
RANK_THRESHOLD = int(os.getenv("RANK_THRESHOLD", "1"))


def cost_order(df) -> np.ndarray:
    """Arm indices, cheapest to dearest, by measured mean latency."""
    return np.argsort(df[LATENCY_COLS].to_numpy().mean(axis=0))


def add_baseline_label(df, rank_threshold: int = RANK_THRESHOLD, verbose: bool = True):
    """THEIR RULE: the cheapest arm that ranked a relevant doc at or above `rank_threshold`.

    MRR = 1 / (rank of first relevant doc), or 0 if none was retrieved, so
    `rank <= T`  <=>  `mrr >= 1/T`. That lets us honour their rank threshold exactly rather
    than approximating it.
    """
    mrr = df[[f"{a}_mrr" for a in ARM_NAMES]].to_numpy()
    success = mrr >= (1.0 / rank_threshold) - 1e-9

    order = cost_order(df)
    label = np.full(len(df), -1, dtype=int)
    for arm in order:                              # cheapest -> dearest, take the first hit
        label[(label == -1) & success[:, arm]] = arm

    hopeless = label == -1
    label[hopeless] = order[0]                     # our deviation: cheapest, not most expensive
    df[BASELINE_LABEL_COL] = label

    if verbose:
        print(f"  T={rank_threshold} | cost order: {[ARM_NAMES[i] for i in order]}")
        print(f"  no arm succeeds: {hopeless.sum():,} ({hopeless.mean():.1%}) "
              f"-> labelled {ARM_NAMES[order[0]]}")
        share = np.bincount(label, minlength=len(ARM_NAMES)) / len(label) * 100
        print("  " + "  ".join(f"{ARM_NAMES[i]}:{share[i]:.1f}%" for i in order))
    return df


def label_ceiling(df):
    """What a router scores if it follows this label PERFECTLY -- the baseline's oracle.

    This is the number that exposes the cost of hard labels: the rule is blind to *where* in
    the top-k the gold doc landed, so a cheap arm at rank 5 (nDCG ~0.39) outranks an expensive
    arm at rank 1 (nDCG 1.0).
    """
    rows = np.arange(len(df))
    sel = df[BASELINE_LABEL_COL].to_numpy()
    return {
        "ndcg": df[NDCG_COLS].to_numpy()[rows, sel].mean(),
        "latency": df[LATENCY_COLS].to_numpy()[rows, sel].mean(),
    }


def hard_ce_loss(logits, labels):
    """THEIR LOSS: cross-entropy onto the single winner.

    No sample mask, unlike ours: their rule assigns all-fail queries a real label (the cheapest
    arm), so those are ordinary training examples here. That asymmetry is inherent to the
    method, not a choice we made on its behalf.
    """
    return F.cross_entropy(logits, labels)


class BaselineObjective(Objective):
    name = "baseline"
    label_dtype = torch.long          # a class index, not a reward vector

    def add_targets(self, df):
        return add_baseline_label(df)

    def targets(self, df):
        return df[BASELINE_LABEL_COL].astype(int).tolist()

    def loss(self, logits, labels):
        return hard_ce_loss(logits, labels)

    def eval_matrix(self, df) -> np.ndarray:
        # No reward exists for this method, so the best epoch is picked on the real retrieval
        # metric: nDCG@5 of whichever arm the model chose. Outcome-based and method-neutral.
        return df[NDCG_COLS].to_numpy()


# ---------------------------------------------------------------------------------------
# THEIR frontier knob: a probability cutoff. Nothing more.
#
# In their 2-class setting the rule is simply
#
#     p(needs the expensive retriever) > tau  ->  dense,  else sparse
#
# Mapped onto ours, with BM25 as the cheap (sparse) arm and every other pipeline as the
# expensive (dense) side:
#
#     p(BM25) >= tau  ->  BM25
#     p(BM25) <  tau  ->  whichever expensive arm the classifier ranks highest
#
# tau = 0 sends everything to BM25; tau = 1 sends everything to an expensive arm. The fraction
# that ends up expensive is their "budget", which is just this cutoff reported in a more useful
# unit. Note what is absent: no latency values, no cost weights. Cost only appears when the
# result is PLOTTED. That is the difference from RetrievalRouter, whose lambda puts cost inside
# training target.
# ---------------------------------------------------------------------------------------
PROB_COLS = [f"pred_prob_{a}" for a in ARM_NAMES]
CHEAP_ARM = "BM25"
DEFAULT_TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def select_at_tau(df, tau: float) -> np.ndarray:
    """Their rule: keep the cheap arm unless the classifier doubts it enough."""
    probs = df[PROB_COLS].to_numpy()
    cheap = ARM_NAMES.index(CHEAP_ARM)

    expensive = [i for i in range(len(ARM_NAMES)) if i != cheap]
    best_expensive = np.array(expensive)[np.argmax(probs[:, expensive], axis=1)]

    return np.where(probs[:, cheap] >= tau, cheap, best_expensive)


def sweep_tau(df, taus=None) -> pd.DataFrame:
    """One (nDCG, latency) operating point per cutoff -- the baseline's frontier.

    `budget` is the fraction of queries that ended up on an expensive arm, i.e. the same knob
    expressed the way their paper reports it.
    """
    taus = DEFAULT_TAUS if taus is None else taus
    ndcg, latency = df[NDCG_COLS].to_numpy(), df[LATENCY_COLS].to_numpy()
    rows = np.arange(len(df))
    cheap = ARM_NAMES.index(CHEAP_ARM)

    out = []
    for tau in taus:
        sel = select_at_tau(df, tau)
        share = np.bincount(sel, minlength=len(ARM_NAMES)) / len(sel) * 100
        out.append({
            "tau": tau,
            "budget": float((sel != cheap).mean()),      # fraction routed to an expensive arm
            "ndcg": ndcg[rows, sel].mean(),
            "latency": latency[rows, sel].mean(),
            **{ARM_NAMES[i]: share[i] for i in range(len(ARM_NAMES))},
        })
    return pd.DataFrame(out)


def random_allocation(df, budgets=None, seed: int = 42, n_trials: int = 20) -> pd.DataFrame:
    """THEIR OWN CONTROL, and the actual claim of their paper.

    Their headline result is not "the classifier beats sparse" -- it is that *at the same
    budget*, the classifier escalates a BETTER SUBSET of queries than random selection does
    (recall 0.95 vs 0.91 at a 50% dense budget).

    Reproduced here: hold the budget and the escalation target fixed, and randomise only WHICH
    queries escalate. So a randomly chosen `b` fraction is sent to the very arm the classifier
    would have chosen for it, and the rest go to BM25. The ONLY difference from sweep_tau is
    which queries get the money -- which is exactly the ablation they run.

    Averaged over n_trials to keep the comparison from riding on one lucky draw.
    """
    budgets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if budgets is None else budgets
    probs = df[PROB_COLS].to_numpy()
    ndcg, latency = df[NDCG_COLS].to_numpy(), df[LATENCY_COLS].to_numpy()
    rows = np.arange(len(df))
    cheap = ARM_NAMES.index(CHEAP_ARM)

    # Each query's escalation target = the expensive arm the classifier ranks highest. Held
    # fixed, so the classifier gets no advantage here beyond *which* queries are picked.
    expensive = [i for i in range(len(ARM_NAMES)) if i != cheap]
    target = np.array(expensive)[np.argmax(probs[:, expensive], axis=1)]

    rng = np.random.default_rng(seed)
    out = []
    for b in budgets:
        n = int(round(b * len(df)))
        nd, lat = [], []
        for _ in range(n_trials):
            sel = np.full(len(df), cheap)
            if n:
                picked = rng.choice(len(df), size=n, replace=False)   # random SUBSET, same targets
                sel[picked] = target[picked]
            nd.append(ndcg[rows, sel].mean())
            lat.append(latency[rows, sel].mean())
        out.append({"budget": b, "ndcg": float(np.mean(nd)), "latency": float(np.mean(lat))})
    return pd.DataFrame(out)
