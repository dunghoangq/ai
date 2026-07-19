from sentence_transformers import SentenceTransformer
from .embedding_model import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    