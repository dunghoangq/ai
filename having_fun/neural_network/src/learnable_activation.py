from abc import ABC, abstractmethod

class LearnableActivation(ABC):

    def __init__(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def forward(self, Z):
        return NotImplementedError
    
    @abstractmethod
    def backward(self, dA):
        return NotImplementedError
    
    @abstractmethod
    def update(self, lr):
        return NotImplementedError