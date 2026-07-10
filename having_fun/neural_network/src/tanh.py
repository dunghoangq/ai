import numpy as np
from .activation import Activation

class Tanh(Activation):

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    
    def backward(self, dA):
        return dA * (1 - self.A ** 2)