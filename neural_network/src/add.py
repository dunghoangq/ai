import numpy as np
from .node import Node

class Add:

    def forward(self, x, y):
        self.x = x
        self.y = y
        return Node(x.value + y.value, parents = (x, y), op="add")
    
    def backward(self, grad_output):
        grad_x = grad_output * 1.0
        grad_y = grad_output * 1.0
        return grad_x, grad_y