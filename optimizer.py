import torch

class SGD(object):
    """Implements stochastic gradient descent"""
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.lr = lr

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        self.params = params

    def step(self):
        """Performs a single optimization step."""
        param_tmp = []
        for (p, grad) in self.params:
            p -= (self.lr * grad)
            param_tmp.append((p, grad))
        self.params = param_tmp

    def zero_grad(self):
        """Clears the gradients of all Tensors."""
        param_tmp = []
        for (p, grad) in self.params:
            grad = 0
            param_tmp.append((p, grad))
        self.params = param_tmp
