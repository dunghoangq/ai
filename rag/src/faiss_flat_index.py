from .vector_index import VectorIndex
import faiss

class FlatIndex(VectorIndex):

    def __init__(self):
        super().__init__()
    

    def build(self, embeddings):
        """
        USAGE:
        index = FlatIndex()
        index.build(embeddings)
        """
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    @property
    def dimension(self):
        return self.index.d