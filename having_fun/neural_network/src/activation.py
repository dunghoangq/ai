from abc import ABC, abstractmethod

class Activation(ABC):
    """
    Base Activation.
    """

    def __init__(self):
        ...
    
    @abstractmethod
    def forward(self, Z):
        return NotImplementedError
    
    @abstractmethod
    def backward(self, dA):
        return NotImplementedError