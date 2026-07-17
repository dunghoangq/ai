import numpy as np
from .activation import Activation

class GELU(Activation):

    def forward(self, Z):
        self.Z = Z
        self.A = (.5*Z*(
            1 + np.tanh(np.sqrt(2/np.pi) * (Z + .044715 * Z**3))
        ))

        return self.A
    
    def backward(self, dA):
        eps = 1e-5
        grad = (self.forward(self.Z + eps) - self.forward(self.Z - eps))/(2*eps)
        return dA * grad