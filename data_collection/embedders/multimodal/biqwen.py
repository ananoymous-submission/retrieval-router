import torch
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory


class BiQwen2_5Embedder(BaseEmbedder):
    """BiQwen2.5 embedder for multimodal tasks."""

    def _initialize_model(self):
        """Initialize the BiQwen2.5 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BiQwen2_5.from_pretrained(
            "nomic-ai/nomic-embed-multimodal-7b",
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
            device_map=self.device,
        ).to(self.device).eval()
    
        self.processor = BiQwen2_5_Processor.from_pretrained("nomic-ai/nomic-embed-multimodal-7b")

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
EmbedderFactory.register_embedder("biqwen", BiQwen2_5Embedder)
