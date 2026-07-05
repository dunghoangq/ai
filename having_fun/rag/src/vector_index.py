import faiss
from abc import ABC, abstractmethod

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
        return self.index.search(query, k)
    
    def save(self, path: str):
        """
        Save the FAISS index.
        """
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """
        Load a FAISS index.
        """
        self.index = faiss.read_index(path)