from abc import ABC, abstractmethod

class Activation(ABC):
    """
    Base Activation.
    """

    def __init__(self):
        self.Z = None
        self.A = None
    
    @abstractmethod
    def forward(self, Z):
        return NotImplementedError
    
    @abstractmethod
    def backward(self, dA):
        return NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)