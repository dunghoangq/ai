import numpy as np
from .activation import Activation

class SoftMax(Activation):
    """
    USAGE: SoftMax()(x)
    """

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)