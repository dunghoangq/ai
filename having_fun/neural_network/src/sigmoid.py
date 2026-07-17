import numpy as np
from .activation import Activation
from .layer import Layer

class Sigmoid(Activation):
    """
    USAGE: Sigmoid()(x)
    """

    # def __call__(self, x):
    #     x = np.asarray(x, dtype=np.float64)

    #     positive = x >= 0
    #     negative = ~positive

    #     result = np.empty_like(x)

    #     result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    #     exp_x = np.exp(x[negative])
    #     result[negative] = exp_x / (1.0 + exp_x)

    #     return result

    def forward(self, Z):
        self.Z = Z
        positive = Z >= 0
        self.A = np.empty_like(Z)

        self.A[positive] = 1 / (1 + np.exp(-Z[positive]))
        exp_Z = np.exp(Z[~positive])
        self.A[~positive] = exp_Z / (1 + exp_Z)

        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1 - self.A)