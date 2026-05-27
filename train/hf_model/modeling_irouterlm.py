"""IRouterLM Model."""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen3Model
from .configuration_irouterlm import IRouterLMConfig


class IRouterLMModel(PreTrainedModel):
    """RAG Strategy Router - classifies queries into optimal retrieval strategies."""
    config_class = IRouterLMConfig
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: IRouterLMConfig):
        super().__init__(config)
        self.transformer = Qwen3Model.from_pretrained(config.base_model_name, trust_remote_code=True)
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
                "strategy_names": [self.config.strategy_names[p.item()] for p in preds]}
