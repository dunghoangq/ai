import numpy as np
from .layer import Layer

class Linear(Layer):

    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) *.01 # smaller weights make early training more stable.
        self.b = np.zeros(out_features)

        # To reuse Numerical Gradient, we can count how often each layer is visited
        self.forward_calls = 0
        self.backward_calls = 0

    def forward(self, X):
        """
        Pytorch stores weight as (out_features, in features).
        So it returns:
            X @ self.W.T + self.b
        """
        X = np.asarray(X, dtype=np.float64)
        self.forward_calls += 1
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dZ):
        """
        dZ = dL_dZ from upstream

        Returns:
            dX
        """
        N = self.X.shape[0]

        self.backward_calls += 1
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        dX = dZ @ self.W.T
        return dX

    def __call__(self, X):
        return self.forward(X)