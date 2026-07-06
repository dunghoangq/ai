import faiss
from abc import ABC, abstractmethod
import numpy as np

class VectorIndex(ABC):
    """
    Not instantiated directly, requires subclasses.
    Applied to FAISS Index family:
        - FlatIndex
        - IVFIndex
        - HNSWIndex
        - IVFPQIndex
    """

    def __init__(self):
        self.index = None

    @property
    @abstractmethod
    def dimension(self):
        raise NotImplementedError
    
    @abstractmethod
    def build(self, embeddings):
        """
        Implemented by subclass.
        Each subclass is enforced to implement build().
        """
        raise NotImplementedError

    def add(self, embeddings):
        """
        Add vectors to index.
        """
        self.index.add(embeddings)

    def search(self, query, k):
        """
        Search query for the k nearest neighbours.

        Returns:
            - D (np.ndarray): Distances
            - I (np.ndarray): Indices of retrieved vectors.
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, ids = self.index.search(query, k)

        return ids[0], scores[0]
    
    def save(self, path: str):
        """
        Save the FAISS index.
        """
        if self.index is None:
            raise ValueError("Index has not been built.")
        faiss.write_index(self.index, str(path))
    
    @classmethod
    def load(cls, path: str):
        """
        Load a FAISS index.
        """
        obj = cls()
        obj.index = faiss.read_index(str(path))
        return obj