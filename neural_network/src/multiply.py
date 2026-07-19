import numpy as np
from .node import Node

class Multiply:

    def forward(self, x, y):
        self.x = x
        self.y = y
        return Node(x.value * y.value, parents=(x, y), op="mul")
    
    def backward(self, grad_output):
        grad_x = grad_output * self.y.value
        grad_y = grad_output * self.x.value
        return grad_x, grad_y