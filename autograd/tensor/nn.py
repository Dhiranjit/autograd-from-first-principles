from autograd.tensor.engine import Tensor
import numpy as np


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.1)
        self.b = Tensor(np.zeros((1, out_features))) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"


class ReLU(Module):
    def __call__(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)


class MLP(Module):
    def __init__(self, in_features, layer_sizes, activation=ReLU):
        layers = []
        sizes = [in_features] + layer_sizes

        for i in range(len(layer_sizes)):
            layers.append(Linear(sizes[i], sizes[i+1]))
            if i != len(layer_sizes) - 1:
                layers.append(activation())

        self.model = Sequential(*layers)

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self):
        return repr(self.model)
    

class CrossEntropyLoss(Module):
    def __call__(self, logits, targets):
        """
        logits: Tensor of shape (batch_size, num_classes)
        targets: numpy array or list of class indices (batch_size,)
        """

        batch_size, num_classes = logits.data.shape

        # ---- Stable softmax ----
        max_logits = Tensor(
            logits.data.max(axis=1, keepdims=True),
            requires_grad=False
        )

        shifted = logits - max_logits
        exp = shifted.exp()
        sum_exp = exp.sum(axis=1, keepdims=True)
        probs = exp / sum_exp

        # ---- Pick correct class probabilities ----
        # Create one-hot manually (no gradient needed)
        one_hot = np.zeros_like(logits.data)
        one_hot[np.arange(batch_size), targets] = 1.0
        one_hot = Tensor(one_hot, requires_grad=False)

        # ---- Cross entropy ----
        log_probs = probs.log()
        loss = -(one_hot * log_probs).sum(axis=1).mean()

        return loss

    def __repr__(self):
        return "CrossEntropyLoss()"