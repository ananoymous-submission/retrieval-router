import torch
from pylate import models
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory

#PYTHONPATH=. python -m data_collection.embedders.text.colbert 

class ColbertEmbedder(BaseEmbedder):
    """Jina-ColBERT-v2 embedder using PyLate for local multi-vector text embedding."""

    def __init__(self, config: EmbedderConfig):
        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(config)

    def _initialize_model(self):
        """Initialize the Jina-ColBERT-v2 model using PyLate."""

        # Load Jina ColBERT-v2 through PyLate (trust_remote_code is required)
        self.model = models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
            document_length=8192,
        ).to(self.device)

    async def embed_document(self, document: str):
        """Embed a single document."""
        return self.model.encode(document, is_query=False)

    async def embed_query(self, query: str):
        """Embed a single query."""
        return self.model.encode(query, is_query=True)


# Register embedder with factory
EmbedderFactory.register_embedder("colbert", ColbertEmbedder)


if __name__ == "__main__":
    import asyncio

    async def main():
        print("Running ColBERT normalization test...\n")
        config = EmbedderConfig(model_name="colbert")
        embedder = ColbertEmbedder(config)
        embedder._initialize_model()

        test_text = "this is a test"
        doc_emb = await embedder.embed_document(test_text)
        qry_emb = await embedder.embed_query(test_text)

        # Compute average L2 norm for both
        def avg_norm(x):
            t = torch.tensor(x, dtype=torch.float32)
            return torch.norm(t, dim=-1).mean().item()

        print(f"Document mean L2 norm: {avg_norm(doc_emb):.4f}")
        print(f"Query mean L2 norm:    {avg_norm(qry_emb):.4f}")

    asyncio.run(main())
