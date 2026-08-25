import torch
import torch.nn as nn


def build_model(num_labels: int):
    """Construct the shared architecture: Qwen3 + LoRA + mean-pool + linear head.

    One builder, used by training and by every method's inference, so the baseline provably
    gets the same network as RetrievalRouter rather than a hand-copied lookalike.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModel

    from train.config import (
        MODEL_NAME, DTYPE, device, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    )

    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    ))
    model = QwenForSoftClassification(
        base_model=base,
        num_labels=num_labels,
        hidden_size=base.config.hidden_size,
        dropout=0.1,
    ).to(device)
    return model


class QwenForSoftClassification(nn.Module):
    """Qwen3 + mean pooling + a linear head over the arms.

    Deliberately logits-only: the loss lives in an Objective, not in the model. That is what
    lets RetrievalRouter and the baselines share this exact architecture while
    differing solely in their training signal.
    """

    def __init__(self, base_model, num_labels: int, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Mean pooling over the sequence, ignoring padding.
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits}
