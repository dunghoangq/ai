import numpy as np
from .kmeans import KMeans

class ProductQuantiser:

    def __init__(self, m: int, k: int):
        '''
        Input:
        - m := number of subspaces.
        - k := number of centroids per subspace, as in K-Means.
        - subvector_dim := computed as d // m during training.
        - codebooks := a list of NumPy arrays, one per subspace. Each array has shape (k, subvector_dim).
        '''
        self.m = m
        self.k = k
        self.subvector_dim = None
        self.codebooks = []

    def fit(self, vectors: np.ndarray):
        # Set dims
        d = vectors.shape[1]
        if d % self.m != 0:
            raise ValueError("Embedding dimension must be divisible by m.")
        self.subvector_dim = d // self.m
        self.codebooks = []

        # Split each vector into m subvectors
        # Since every block is the same size, use split instead of array_split.
        subspace_vectors = np.split(vectors, self.m, axis=1)

        # For each subspace
        for subspace_vector in subspace_vectors:
            # K-Means
            kmeans = KMeans(self.k)
            kmeans.fit(subspace_vector)

            # Store the centroids as a codebook
            # don't use extend here because it'd treat rows individually (list of centroids), instead of list of codebooks.
            self.codebooks.append(kmeans.centroids)

    def encode(self, vectors: np.ndarray):
        '''
        - For every new vector arrives, for every subspace
        - Nearest centroid
        - Store centroid index

        Output:
            - [2 1 0] := subspace 0: centroid 2, subspace 1: centroid 1, subspace 2: centroid 0...
        '''
        # Split vectors
        subspaces = np.split(vectors, self.m, axis=1)
        encoded = np.empty((len(vectors), self.m), dtype=np.uint8)

        # For each subspace
        for i, (subspace_vectors, codebook) in enumerate(zip(subspaces, self.codebooks)):
            # Get nearest centroid
            dot_product = subspace_vectors @ codebook.T
            subvector_norms = np.linalg.norm(subspace_vectors, axis=1)
            codebook_norms = np.linalg.norm(codebook, axis=1)

            scores = dot_product / (subspace_vectors[:, None] * codebook_norms[None, :])
        
            # Store result
            encoded[:, i] = np.argmax(scores, axis=1)
        
        return encoded

    
    def decode(self, encoded: np.ndarray):
        '''
        Input:
            - encoded: list of centroid indexes mapped to subspaces.
            [2 1 0]
        
        Output:
            - Concatenated decoded subspaces.
            reconstruct codebook[0][2] + codebook[1][1] + codebook[2][0]
        
        '''
        decoded = []

        for code in encoded:
            subvectors = []

            for subspace, centroid_index in enumerate(code):
                subvectors.append(self.codebooks[subspace][centroid_index])
            decoded.append(np.concatenate(subvectors))
        return np.array(decoded)