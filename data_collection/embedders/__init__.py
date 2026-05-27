"""
Embedder Framework

This package provides a generalized interface for working with different embedders,
allowing easy switching between text and multimodal embedding models.
"""

from .base_embedder import (
    BaseEmbedder,
    EmbedderConfig,
    EmbedderFactory
)

# Lazy import - embedders are registered when first accessed
# This avoids loading heavy dependencies until actually needed

def _ensure_embedders_registered():
    """Register embedders on first access to avoid heavy startup imports."""
    if not hasattr(_ensure_embedders_registered, '_registered'):
        # Only import what we actually need based on environment
        import os
        pipeline = os.getenv("PIPELINE_NAME", "")
        two_stage_mode = os.getenv("TWO_STAGE_MODE", "multimodal")

        if "MULTIMODAL" in pipeline or (pipeline == "TWO-STAGE" and two_stage_mode == "multimodal"):
            # Import specific multimodal embedders
            from .multimodal.colqwen import ColQwen2_5Embedder
            from .multimodal.biqwen import BiQwen2_5Embedder
            from .multimodal.voyage import VoyageMultimodalEmbedder
        if "TEXT" in pipeline or (pipeline == "TWO-STAGE" and two_stage_mode == "text"):
            # Import specific text embedders
            from .text.linq import LinqEmbedder
            from .text.voyage import VoyageTextEmbedder
            from .text.colbert import ColbertEmbedder
        _ensure_embedders_registered._registered = True

# Monkey patch the factory to ensure registration on first use
_original_create_embedder = EmbedderFactory.create_embedder

@classmethod
def _lazy_create_embedder(cls, name: str, config):
    _ensure_embedders_registered()
    return _original_create_embedder(name, config)

EmbedderFactory.create_embedder = _lazy_create_embedder

__all__ = [
    'BaseEmbedder',
    'EmbedderConfig',
    'EmbedderFactory'
] 