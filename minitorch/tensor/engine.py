import numpy as np

class Tensor:
    def __init__(self, data, _parents=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
    
    @property
    def shape(self):
        return self.data.shape  
    
   
    @property  
    def size(self):
        return self.data.size 


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if out.grad is None:
                return
                
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad_self = out.grad # (10, 5)

                # Case 1: Handle different number of dimensions
                ndims_added = grad_self.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad_self = grad_self.sum(axis=0) # Along the columns

                # Case 2: 
                for i in range(self.data.ndim): # i = 0, 1
                    if self.data.shape[i] == 1 and grad_self.shape[i] > 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                
                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad # (10, 5)

                # Case 1
                ndims_added = grad_other.ndim - other.data.ndim
                for i in range(ndims_added):  # i = 0
                    grad_other = grad_other.sum(axis=0) 

                # Case 2
                for i in range(other.data.ndim):  # i = 0
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other
                
        out._backward = _backward
        return out
    

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)


    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += -out.grad
        
        out._backward = _backward
        return out 
    

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.requires_grad and out.grad is not None:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * other.data
            
            if other.requires_grad and out.grad is not None:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

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
    

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self, ), 'getitem')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad[idx] += out.grad
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __radd__(self, other):
        return self + other


    def log(self):
        eps = 1e-8
        out = Tensor(np.log(np.clip(self.data, eps, 1.0)), (self,), 'log')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Derivative of log(x) is 1/x
                self.grad += out.grad / np.clip(self.data, eps, 1.0)
        
        out._backward = _backward
        return out


    def softmax(self):
        # Numerically stable softmax
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        out = Tensor(probs, (self,), 'softmax')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Softmax gradient: softmax * (grad - sum(softmax * grad))
                softmax_grad = out.data * (out.grad - np.sum(out.data * out.grad, axis=-1, keepdims=True))
                self.grad += softmax_grad
        
        out._backward = _backward
        return out
    
    


    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Derivative of ReLU: 1 if x > 0, else 0
                self.grad += out.grad * (self.data > 0).astype(np.float32)
        
        out._backward = _backward
        return out


    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
                self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out

    @property
    def T(self):
        """Property version of transpose for .T syntax"""
        out = Tensor(self.data.T, (self,), 'transpose')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Gradient of transpose is transpose of gradient
                self.grad += out.grad.T
        
        out._backward = _backward
        return out



    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Ensure out.grad is properly broadcast
                grad_array = np.asarray(out.grad)
                if grad_array.ndim == 0:
                    grad_val = float(grad_array)
                else:
                    grad_val = float(grad_array.item())
                self.grad += np.full_like(self.data, grad_val / self.data.size)
                
        out._backward = _backward
        return out


    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)


    @staticmethod
    def randn(*shape, requires_grad =True):
        data = np.random.randn(*shape).astype(np.float32) * 0.1
        return Tensor(data, requires_grad=requires_grad)
    

    ### String representation of the Tensor
    def __repr__(self):
        return f"{repr(self.data)}, requires_grad={self.requires_grad}"
    
    def backward(self):
        # Build topological order of computation graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output (root)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # Backpropagate through computation graph
        for node in reversed(topo):
            node._backward()
    

class Losses:
    @staticmethod
    def cross_entropy(logits, targets):
        """
        Combined softmax + cross-entropy loss for numerical stability.
        
        Args:
            logits: Raw model outputs (shape: [batch_size, num_classes])
            targets: Either class indices (shape: [batch_size]) or one-hot vectors (shape: [batch_size, num_classes])
        
        Returns:
            loss: Scalar tensor containing the average cross-entropy loss
        """
        # Convert targets to one-hot if they're class indices
        if targets.data.ndim == 1:
            num_classes = logits.data.shape[-1]
            batch_size = targets.data.shape[0]
            y_one_hot = np.zeros((batch_size, num_classes))
            y_one_hot[np.arange(batch_size), targets.data.astype(int)] = 1
            targets_tensor = Tensor(y_one_hot, requires_grad=False)
        else:
            targets_tensor = targets
        
        # Numerically stable softmax computation
        shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        softmax_probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        # Cross-entropy loss: -sum(y * log(p))
        eps = 1e-8
        log_probs = np.log(np.clip(softmax_probs, eps, 1.0))
        loss_per_sample = -np.sum(targets_tensor.data * log_probs, axis=-1)
        avg_loss = np.mean(loss_per_sample)
        
        out = Tensor(avg_loss, (logits,), 'cross_entropy_loss')
        
        def _backward():
            if logits.requires_grad:
                if logits.grad is None:
                    logits.grad = np.zeros_like(logits.data)
                
                # Simplified gradient: (softmax(x) - y) / batch_size
                batch_size = logits.data.shape[0]
                grad = (softmax_probs - targets_tensor.data) / batch_size
                logits.grad += out.grad * grad
        
        out._backward = _backward
        return out
    
    @staticmethod
    def mse(predictions, targets):
        """Mean Squared Error loss"""
        diff = predictions - targets
        return (diff * diff).mean()
    
    @staticmethod
    def binary_cross_entropy(predictions, targets):
        """Binary cross-entropy for binary classification"""
        eps = 1e-8
        clipped_preds = Tensor(np.clip(predictions.data, eps, 1 - eps), (predictions,))
        return -(targets * clipped_preds.log() + (1 - targets) * (1 - clipped_preds).log()).mean()

# Usage examples:
# loss = Losses.cross_entropy(logits, targets)
# loss = Losses.mse(predictions, targets)
