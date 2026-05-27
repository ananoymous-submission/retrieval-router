import torch
import torch.nn.functional as F
from train.config import TEMPERATURE


def weighted_kl_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted KL divergence loss for soft label learning.

    Targets are produced by a temperature-scaled softmax over the reward
    vector, which sharpens the distribution so small reward gaps (e.g.,
    latency-only differences on easy queries) produce useful gradient.

    Samples are weighted by their max reward value, so zero-reward samples
    contribute zero gradient (uninformative queries are ignored).

    Args:
        logits: Model output logits, shape (batch_size, num_classes)
        labels: Soft labels (reward values), shape (batch_size, num_classes)

    Returns:
        Weighted mean loss (scalar tensor)
    """
    labels_normalized = F.softmax(labels / TEMPERATURE, dim=-1)

    log_probs = F.log_softmax(logits, dim=-1)

    # Calculate KL Divergence (Cross Entropy) per sample
    sample_losses = -(labels_normalized * log_probs).sum(dim=-1)

    # Weight the loss by max reward - if all rewards were 0, sample contributes 0 gradient
    sample_weights = labels.max(dim=-1)[0]

    # Weighted mean loss
    loss = (sample_losses * sample_weights).mean()

    return loss
