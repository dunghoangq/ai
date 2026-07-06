from .vector_store import VectorStore

class Retriever:

    def __init__(self, vector_store: VectorStore, bm25=None, rank_fusion=None, reranker=None, compressor=None, score_threshold=0.6):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rank_fusion = rank_fusion
        self.reranker = reranker
        self.compressor = compressor
        self.score_threshold = score_threshold

    def retrieve(self, query, k=5):
        if self.bm25:
            dense = self.vector_store.search(query, 20)
            sparse = self.bm25.search(query, 20)
            candidates = self.rank_fusion.fuse(dense, sparse)
        else:
            candidates = self.vector_store.search(query, 20)

        if self.reranker:
            candidates = self.reranker.rank(query, candidates)

        candidates = [
            r for r in candidates if r.score >= self.score_threshold
        ]

        if self.compressor:
            candidates = self.compressor.compress(query, candidates)
    
        return candidates[:k]