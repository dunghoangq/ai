import numpy as np
from .activation import Activation

class Swish(Activation):

    def forward(self, Z):
        self.Z = Z
        self.sigmoid = 1 / (1 + np.exp(-Z))
        return Z*self.sigmoid
    
    def backward(self, dA):
        grad = self.sigmoid + self.Z * self.sigmoid*(1 - self.sigmoid)
        return dA*grad