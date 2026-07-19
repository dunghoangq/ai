from abc import ABC, abstractmethod
import numpy as np

class EmbeddingModel(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple documents.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        np.ndarray
            Shape = (n_texts, dimension)
        """
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.

        Returns
        -------
        np.ndarray
            Shape = (dimension,)
        """
        return self.embed([query])[0]