"""RetrievalRouter model."""
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, Qwen3Model
from .configuration_retrievalrouter import RetrievalRouterConfig


class RetrievalRouterModel(PreTrainedModel):
    """RetrievalRouter model that classifies queries into retrieval pipelines."""
    config_class = RetrievalRouterConfig
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: RetrievalRouterConfig):
        super().__init__(config)
        # Build the base architecture only; the merged base weights are loaded from this
        # checkpoint's model.safetensors by from_pretrained. Calling Qwen3Model.from_pretrained
        # here breaks under the meta-device init that from_pretrained uses.
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        self.transformer = Qwen3Model(base_config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = self._compute_loss(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def _compute_loss(self, logits, labels):
        labels_norm = labels / (labels.sum(-1, keepdim=True) + 1e-8)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        losses = -(labels_norm * log_probs).sum(-1)
        return (losses * labels.max(-1)[0]).mean()

    def predict(self, input_ids, attention_mask=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return {"predictions": preds, "probabilities": probs, 
                "pipeline_names": [self.config.pipeline_names[p.item()] for p in preds]}
