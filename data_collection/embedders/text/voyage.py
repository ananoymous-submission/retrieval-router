import voyageai
from ..base_embedder import BaseEmbedder, EmbedderFactory

class VoyageTextEmbedder(BaseEmbedder):

    def _initialize_model(self):
        self.client = voyageai.Client()
        self.model_name = "voyage-3-large"

    async def embed_document(self, document):
        result = self.client.embed(
            texts=[document],
            model=self.model_name,
            input_type="document"
        )
        return result.embeddings[0]

    async def embed_query(self, query):
        result = self.client.embed(
            texts=[query],
            model=self.model_name,
            input_type="query"
        )
        return result.embeddings[0]


EmbedderFactory.register_embedder("voyage_text", VoyageTextEmbedder)
