import torch
import torch.nn as nn

from train.loss import weighted_kl_loss


class QwenForSoftClassification(nn.Module):
    """
    Qwen3 model with classification head for soft label learning.
    Uses mean pooling and weighted KL divergence loss.
    """

    def __init__(self, base_model, num_labels: int, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Mean pooling over sequence dimension
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask

        # Classification head
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = weighted_kl_loss(logits, labels)

        return {"loss": loss, "logits": logits}
