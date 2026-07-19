import numpy as np

class Node:

    def __init__(self, value, parents=(), op=""):
        self.value = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.op = op
        self.grad = np.zeros_like(self.value)

    def add(a, b):
        return Node(
            value=a.value + b.value,
            parents=(a, b),
            op="add"
        )

    def mul(a, b):
        return Node(
            values=a.value * b.value,
            parents=(a, b),
            op="mul"
        )
    
    def square(a):
        return Node(
            value=a.value ** 2,
            parents=(a,),
            op="square"
        )

    def __repr__(self):
        return f"Node(value={self.value}, op='self.op')"