"""RetrievalRouter's reward-vector training objective.

Target: the full per-query reward vector r_i = (1-L)*nDCG_i + L*(1 - normalised_latency_i),
turned into a target distribution with softmax(r / T). The model therefore learns *how much* better
each arm is, not merely which one wins.

Loss: KL(target || pi), computed as cross-entropy against the target distribution. Samples with
an all-zero reward vector are masked out. At lambda greater than zero, the efficiency term keeps
all-fail queries in training by providing a nonzero signal.
"""
import numpy as np
import torch
import torch.nn.functional as F

from train.config import REWARD_COLS, TEMPERATURE
from train.objectives import Objective


def retrievalrouter_loss(logits, labels):
    """KL onto softmax(reward / T), averaged over queries that have any signal."""
    target_distribution = F.softmax(labels / TEMPERATURE, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target_distribution * log_probs).sum(dim=-1)

    # Exclude only all-zero reward vectors. At lambda > 0, efficiency keeps rewards nonzero.
    mask = (labels.max(dim=-1)[0] > 0).float()
    return (per_sample * mask).sum() / mask.sum().clamp(min=1)


class RetrievalRouterObjective(Objective):
    name = "retrievalrouter"
    label_dtype = torch.float32

    def add_targets(self, df):
        from train.prepare_data import calculate_rewards   # local: avoids a circular import
        return calculate_rewards(df)

    def targets(self, df):
        return [list(r) for r in df[REWARD_COLS].to_numpy()]

    def loss(self, logits, labels):
        return retrievalrouter_loss(logits, labels)

    def eval_matrix(self, df) -> np.ndarray:
        # Best epoch = highest reward-of-chosen-arm, i.e. the quantity we actually optimise.
        return df[REWARD_COLS].to_numpy()
