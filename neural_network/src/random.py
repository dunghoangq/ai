import numpy as np

class RandomNormal:

    def __call__(self, shape):
        return np.random.randn(*shape)