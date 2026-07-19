import numpy as np

class Parameter:
    """
    A trainable tensor
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad.fill(.0)

    def __repr__(self):
        return (f"Parameter(shape={self.data.shape})")