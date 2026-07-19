import numpy as np
from .cosine_similarity import cosine_similarity

class KMeans:
    
    def __init__(self, k: int):
        self.k = k
        self.centroids = None
        self.labels = None
    
    def initialise_centroids(self, vectors: np.ndarray):
        '''
        Select k random centroids.
        '''
        indices = np.random.choice(len(vectors), size=self.k, replace=False)
        self.centroids = vectors[indices]
        return self.centroids


    def assign_clusters(self, vectors: np.ndarray):
        '''
        Assign input vector to a centroid.

        vectors: (m x n)
        centroids: (k x n)

        Each of m vectors cosine with each of k centroids.
        Return the scores (m x n) vectors x (k x n)^T centroids.
        '''
        if self.centroids is None:
            return "No centroids created."
        
        dot_products = vectors @ self.centroids.T               # (m x k)
        vector_norms = np.linalg.norm(vectors, axis=1)          # (m x 1)
        centroid_norms = np.linalg.norm(self.centroids, axis=1) # (k x 1)

        # Since we need the denominator has the shape (m x k)
        # vector_norms and centroid_norms have to be (m x 1) x (1 x k)
        scores = dot_products / (vector_norms[:, None] * centroid_norms[None, :])

        self.labels = np.argmax(scores, axis=1)

    def update_centroids(self, vectors: np.ndarray):
        '''
        Traditional K-Means minimises Euclidean distance and updates centroids with the arithmetic mean. If you want cosine similarity, you'd normally normalise the vectors and centroids (sometimes called spherical K-Means).

        For learning purposes, your implementation is perfectly fine. Just be aware that you're mixing two different clustering objectives.
        '''
        if self.centroids is None or self.labels is None:
            return "No centroids or labels to update!"
        
        for cluster in range(self.k):
            cluster_vectors = vectors[self.labels == cluster]
            if len(cluster_vectors) == 0:
                continue
            self.centroids[cluster] = np.mean(cluster_vectors, axis=0)

    def fit(self, vectors: np.ndarray, max_iterations=100):
        '''
        Once centroids are moved, they change which vectors belong to them.
        We have to do the process again and again till Convergence.
        '''
        self.initialise_centroids(vectors)

        for _ in range(max_iterations):
            self.assign_clusters(vectors)
            old_centroids = self.centroids.copy()
            self.update_centroids(vectors)

            # This is better than old_centroids == self.centroids
            if np.allclose(old_centroids, self.centroids):
                break
