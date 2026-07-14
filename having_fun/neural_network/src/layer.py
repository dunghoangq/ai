from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Base Layer class
    """

    @abstractmethod
    def forward(self, x):
        return NotImplementedError
    
    @abstractmethod
    def backward(self, grad):
        return NotImplementedError