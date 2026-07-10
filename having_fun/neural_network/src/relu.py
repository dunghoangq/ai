import numpy as np
from .activation import Activation

class ReLU(Activation):

    # def __call__(self, x):
    #     x = np.asarray(x, dtype=np.float64)
    #     return np.maximum(0.0, x)
    
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        return dA * (self.Z > 0)