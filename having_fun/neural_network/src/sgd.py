import numpy as np
from .optimiser import Optimiser

class SGD(Optimiser):

    def __init__(self, parameters, lr=.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad