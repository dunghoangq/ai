from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    """
    BM25 has complex mathematics (IDF, TF saturation, document length normalisation). For the purpose of learning at this point, just use API.
    Will build from scratch later.
    """

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def build(self, documents):
        self.documents = documents
        corpus = [doc.text.split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=20):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []

        for idx in top_indices:
            doc = self.documents[idx]
            doc.score = scores[idx]
            results.append(doc)
        
        return results