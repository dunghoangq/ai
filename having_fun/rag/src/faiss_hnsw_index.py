from .vector_index import VectorIndex
import faiss

class HNSWIndex(VectorIndex):
    
    def __init__(self, m=32, efConstruction=200, efSearch=64):
        super().__init__()
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch

    def build(self, embeddings):
        """
        USAGE:
        index = HNSWIndex()
        index.build(embeddings)
        """
        d = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(d, self.m)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch
        self.add(embeddings)

    @property
    def dimension(self):
        return self.index.d