"""IRouterLM Configuration."""
from transformers import PretrainedConfig

STRATEGY_NAMES = ["MULTIMODAL_RERANK", "MULTIMODAL-SINGLE", "TEXT_RERANK", "TEXT-SINGLE"]


class IRouterLMConfig(PretrainedConfig):
    """Configuration for IRouterLM - a RAG strategy router model."""
    model_type = "irouterlm"

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B-Base",
        hidden_size: int = 1024,
        num_labels: int = 4,
        classifier_dropout: float = 0.1,
        strategy_names: list = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.strategy_names = strategy_names or STRATEGY_NAMES
