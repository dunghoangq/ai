import numpy as np

class GradientDescent:
    """
    USAGE:
        optimiser = GradientDescent(.1)

        for step in steps:
            params = optimiser.step(params, grads)
    """

    def __init__(self, learning_rate=.1):
        self.lr = learning_rate

    def step(self, params, grads):
        """
        Input:
            params: numpy array
            grads: numpy array

        Returns
            new params: numpy array
        """
        return params - self.lr * grads