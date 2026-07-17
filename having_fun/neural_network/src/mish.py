import numpy as np
from .activation import Activation

class Mish(Activation):

    def forward(self, Z):
        self.Z = Z
        self.softplus = np.log1p(np.exp(Z))
        self.tanh = np.tanh(self.softplus)
        return x*self.tanh
    
    def backward(self, dA):
        eps = 1e-5

        grad = (self.forward(self.Z + eps) - self.forward(self.Z - eps)) / (2*eps)

        return dA*grad