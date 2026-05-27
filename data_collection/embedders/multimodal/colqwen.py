import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory


class ColQwen2_5Embedder(BaseEmbedder):
    """ColQwen2.5 embedder for multimodal tasks."""

    def _initialize_model(self):
        """Initialize the ColQwen2.5 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ColQwen2_5.from_pretrained(
            "Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map=self.device,
        ).to(self.device).eval()
    
        self.processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/ColQwen2.5-7b-multilingual-v1.0")

    async def embed_document(self, image):
        """Embed a document (image)."""
        with torch.inference_mode():
            batch_image = self.processor.process_images([image]).to(self.device)
            embedding = self.model(**batch_image)
            embedding = embedding.cpu().float().numpy().tolist()

        return embedding[0]

    async def embed_query(self, query):
        """Embed a text query."""
        with torch.inference_mode():
            batch_query = self.processor.process_queries([query]).to(self.device)
            embedding = self.model(**batch_query)
            embedding = embedding.cpu().float().numpy().tolist()

        return embedding[0]


# Register embedder with factory
EmbedderFactory.register_embedder("colqwen", ColQwen2_5Embedder)
