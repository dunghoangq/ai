import numpy as np
from .node import Node

class Square:

    def forward(self, x):
        self.x = x
        return Node(x.value, parents=(x,), op="square")
    
    def backward(self, grad_output):
        grad_x = grad_output * (2 * self.x.value)
        return grad_x