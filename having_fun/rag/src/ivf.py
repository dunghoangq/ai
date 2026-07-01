import numpy as np
from .kmeans import KMeans
from .embedding import cosine_similarity

class Posting:

    def __init__(self, chunk_id, text, embedding):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding

class IVFIndex:

    def __init__(self):
        self.centroids = None       # (k, d)
        self.inverted_lists = {}    # dict[int, list[Posting]]
        self.vectors = None

    def fit(self, vectors: np.ndarray):
        '''
        Run K-Means to return centroids and labels.
        Then build the inverted_lists.
        '''
        kmeans = KMeans()
        kmeans.fit(vectors)
        self.centroids = kmeans.centroids
        labels = kmeans.labels
        
        # Loop for inverted lists
        self.inverted_lists = {i: [] for i in range(len(self.centroids))}
        for idx, label in enumerate(labels):
                self.inverted_lists[label].append(idx)
        self.vectors = vectors


    def search(self, query_embedding: np.array, k: int, nprobe: int):
        '''
        Get the nearest centroids > Retrieve postings > Similarity > Sort > Top-k chunks.

        Input

        - k: top-k vectors
        - nprobe: # centroids to look at
        '''
        nearest_centroids = self._find_nearest_centroids(query_embedding, nprobe)
        candidate_ids = []
        candidate_scores = []
        
        for centroid in nearest_centroids:
            ids = self.inverted_lists[centroid]
            scores = self._cosine_similarity(query_embedding, self.vectors[posting_idx])
            candidate_ids.extend(ids)
            candidate_scores.extend(scores)
        
        candidate_ids = np.concatenate(candidate_ids)
        candidate_scores = np.concatenate(candidate_scores)
        candidate_idx, candidate_values = self._rank_candidates(candidate_scores, k)
        
        return candidate_ids[candidate_idx], candidate_values
        
    
    def _find_nearest_centroids(self, query_embedding: np.array, nprobe: int):
        scores = self._cosine_similarity(query_embedding, self.centroids)
        return np.argpartition(scores, -nprobe)[-nprobe:]
    
    def _cosine_similarity(self, query_embedding: np.ndarray, vectors_embedding: np.ndarray):
        dot_product = query_embedding @ vectors_embedding.T
        query_norms = np.linalg.norm(query_embedding)
        vector_norms = np.linalg.norm(vectors_embedding, axis=1)
        return dot_product / (query_norms * vector_norms)

    def _rank_candidates(self, scores: np.ndarray, k: int):
        '''
        Rank Top-k candidates
        '''
        indices = np.argpartition(scores, -k)[-k:]
        values = scores[indices]
        return indices, values