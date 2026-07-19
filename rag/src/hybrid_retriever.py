class HybridRetriever:

    def __init__(self, vector_store, bm25):
        self.vector_store = vector_store
        self.bm25 = bm25