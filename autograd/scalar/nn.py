import random
from autograd.scalar.engine import Value


def cross_entropy_loss(logits, y_true):
    n = len(logits)

    # Accept a class index or an explicit one-hot vector.
    if isinstance(y_true, int):
        one_hot = [1.0 if i == y_true else 0.0 for i in range(n)]
    else:
        one_hot = [float(v) for v in y_true]

    # Numerically stable softmax: subtract max logit (a constant shift that
    # cancels in softmax but prevents exp() overflow).
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    sum_exp = sum(exp_vals[1:], exp_vals[0])   # Value accumulation
    probs = [e / sum_exp for e in exp_vals]

    # Cross-entropy: -sum over classes where y_i != 0 (avoids log(p) * 0 terms)
    loss = sum(
        (-p.log() * yi for p, yi in zip(probs, one_hot) if yi != 0.0),
        Value(0.0),
    )
    return loss


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron(nin={len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        nin = len(self.neurons[0].w)
        nout = len(self.neurons)
        kind = "ReLU" if self.neurons[0].nonlin else "Linear"
        return f"{kind}Layer(in={nin}, out={nout})"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        lines = ["MLP("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)