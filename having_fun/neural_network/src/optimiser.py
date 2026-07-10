from abc import ABC, abstractmethod

class Optimiser(ABC):
    """
    base class for:
        - SGD
        - Adam
        - RMSProp
        - Momentum
        - AdamW
    """

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
    
    @abstractmethod
    def step(self):
        raise NotImplementedError