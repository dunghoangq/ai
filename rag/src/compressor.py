from abc import ABC, abstractmethod

class Compressor(ABC):
    """
    Base class for contextual compression.

    Takes retrieved documents and returns shorter,
    more relevant versions.
    """ 

    @abstractmethod
    def compress(self, query, documents):
        raise NotImplementedError