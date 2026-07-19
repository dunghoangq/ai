import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)         # numerical value
        self.grad = 0.0                 # accumulated gradient
        self._prev = set(_children)     # parent nodes
        self._op = _op                  # the operation that created it
        self._backward = lambda: None   # how to backprop gradients to parents

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward - _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        """
        DFS - topo sort from input to output.
        Reverse mode:
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()
        """
        topo = []
        visited = set()

        def _build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build(child)
                topo.append(node)

        _build(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()