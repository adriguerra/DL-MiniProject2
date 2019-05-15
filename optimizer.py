import torch
import modules

class SGD(object):
    """Implements stochastic gradient descent"""
    def __init__(self, arch, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.lr = lr

        if not isinstance(arch, modules.Module):
            raise TypeError("Architecture given to the optimizer should be "
                            "an instance of a module, but got " +
                            torch.typename(arch))

        self.arch = arch

    def step(self):
        """Performs a single optimization step."""
        for (p, grad) in self.arch.param():
            p.data -= (self.lr * grad)

    def zero_grad(self):
        """Clears the gradients of all Tensors."""
        for (p, grad) in self.arch.param():
            grad.data = torch.zeros(grad.shape)
