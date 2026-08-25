"""Training signals, one file per approach.

Every routing method in this project shares the same pipeline -- encoder, LoRA, mean pooling,
linear head, seeded split, tokenisation, optimiser, schedule, checkpoint selection, inference.
The ONLY things a method changes are:

    1. how the per-query training target is built from the measured pipeline metrics
    2. what loss is applied to the model's logits given that target

An Objective bundles exactly those two. Selecting one is a config switch (`OBJECTIVE`), so
two methods are identical by construction rather than by inspection -- which is what makes
the comparison between them mean anything.

    retrievalrouter.py  RetrievalRouter: full reward vector + KL
    baseline.py         Arabzadeh et al. (CIKM 2021): hard label + cross-entropy
"""
from abc import ABC, abstractmethod

import numpy as np
import torch


class Objective(ABC):
    """A training signal: how to build the target, and how to score logits against it."""

    #: short identifier; the value of the OBJECTIVE config switch
    name: str

    #: dtype the collated label tensor is cast to before `loss` sees it
    label_dtype: torch.dtype

    @abstractmethod
    def add_targets(self, df):
        """Write this objective's target column(s) onto `df` in place (called by prepare_data)."""

    @abstractmethod
    def targets(self, df):
        """The per-row target the HF Dataset stores as `labels`."""

    @abstractmethod
    def loss(self, logits, labels):
        """Scalar training loss from raw logits and the collated target."""

    @abstractmethod
    def eval_matrix(self, df) -> np.ndarray:
        """(n_queries, n_arms) matrix used to pick the best epoch.

        compute_metrics reports predicted_ndcg = eval_matrix[i, argmax(logits_i)], so this is
        what "best checkpoint" means for the method.
        """


def get_objective(name: str) -> Objective:
    """Resolve the OBJECTIVE config switch to an instance."""
    from train.objectives.baseline import BaselineObjective
    from train.objectives.retrievalrouter import RetrievalRouterObjective

    registry = {
        RetrievalRouterObjective.name: RetrievalRouterObjective,
        BaselineObjective.name: BaselineObjective,
    }
    if name not in registry:
        raise ValueError(f"Unknown OBJECTIVE {name!r}; expected one of {sorted(registry)}")
    return registry[name]()
