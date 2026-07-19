import numpy as np
from .activation import Activation
from .itentity import Identity
from .random import RandomNormal

class Neuron:

    def __init__(self, input_dim, activation: Activation=None, initialiser=None):
        if initialiser is None:
            initialiser = RandomNormal()
        self.W = initialiser(input_dim)
        self.b = np.zeros((1,))
        if activation is None:
            activation = Identity()
        
        self.activation = activation()
        self.cache = {}

    def _validate_input(self, X):
        if X.ndim != 2:
            raise ValueError("Expected input shape (batch_size, input_size)")
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Expected {self.W.shape[0]} features, "
                f"received {X.shape[1]}"
            )

    def linear(self, X):
        return X @ self.W + self.b
    
    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X)
        Z = self.linear(X)
        A = self.activation.forward(Z)
        
        self.cache = {
            "X": X,
            "Z": Z,
            "A": A
        }

        return A
    
    def backward(self, y):
        dL_da = self.cache['A'] - y
        da_dz = self.cache['A'] * (1 - self.cache['A'])
        delta = dL_da * da_dz

        dw = delta * self.cache['X']
        db = delta
        dx = delta * self.W

        return {
            "dw": dw,
            "db": db,
            "dx": dx
        }
    
    def predict(self, X):
        """
        Convert score of x into class with boundary z = 0.
        """
        return int(self.forward(X) >= 0)