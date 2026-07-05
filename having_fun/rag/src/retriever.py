from .vector_store import VectorStore

class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query, k=5):
        return self.vector_store.search(query, k)