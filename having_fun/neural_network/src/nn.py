import numpy as np
from .linear import Linear
from .relu import ReLU
from .mseloss import MSELoss


class NeuralNetwork:

    def __init__(self):
        self.fc1 = Linear(2, 4)
        self.relu = ReLU()
        self.fc2 = Linear(4, 1)
        self.loss = MSELoss()

    def forward(self, X, y):
        h = self.fc1.forward(X)
        h = self.relu.forward(h)
        pred = self.fc2.forward(h)
        loss = self.loss.forward(pred, y)
        return pred, loss

    def backward(self):
        grad = self.loss.backward()
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)