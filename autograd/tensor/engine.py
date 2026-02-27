import numpy as np


def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor():
    def __init__(self, data, requires_grad=True, _prev=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set(_prev)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # --------------------
    # Addition
    # --------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, _prev=(self, other), _op='+')

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += unbroadcast(out.grad, self.data.shape)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # --------------------
    # Subtraction
    # --------------------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # --------------------
    # Multiplication
    # --------------------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, _prev=(self, other), _op='*')

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = other.data * out.grad
                self.grad += unbroadcast(grad_self, self.data.shape)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = self.data * out.grad
                other.grad += unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # --------------------
    # Division
    # --------------------
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1)

    # --------------------
    # Power
    # --------------------
    def __pow__(self, power):
        out = Tensor(self.data ** power, _prev=(self,), _op='pow')

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # --------------------
    # Matrix Multiplication
    # --------------------
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, _prev=(self, other), _op='@')

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out


    # Activations
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _prev=(self,), _op='tanh')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(0, self.data)
        out = Tensor(r, _prev=(self,), _op='relu')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out


    def mean(self):
        out = Tensor(self.data.mean(), _prev=(self,), _op='mean')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (np.ones_like(self.data) / self.data.size) * out.grad

        out._backward = _backward
        return out
    

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, _prev=(self,), _op="exp")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += e * out.grad

        out._backward = _backward
        return out


    def log(self):
        out = Tensor(np.log(self.data), _prev=(self,), _op="log")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out


    # Sum (with axis support)
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     _prev=(self,), _op="sum")

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)

                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out


    # Backward
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()