"""RetrievalRouter configuration."""
from transformers import PretrainedConfig

# Standalone copy of train.config.ARM_NAMES: this module is uploaded to the Hub and loaded
# via trust_remote_code, so it cannot import from the training package.
PIPELINE_NAMES = ["MULTIMODAL_RERANK", "MULTIMODAL-SINGLE", "TEXT_RERANK", "TEXT-SINGLE", "BM25"]


class RetrievalRouterConfig(PretrainedConfig):
    """Configuration for the RetrievalRouter query-aware pipeline selector."""
    model_type = "retrievalrouter"

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B-Base",
        hidden_size: int = 1024,
        num_labels: int = 5,
        classifier_dropout: float = 0.1,
        pipeline_names: list | None = None,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.pipeline_names = pipeline_names or PIPELINE_NAMES
