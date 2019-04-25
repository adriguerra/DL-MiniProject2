import torch

def tanh(x):
    return x.tanh()

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def loss(v, t):
    return (v - t).pow(2).sum()

def dloss(v, t):
    return 2 * (v - t)

def relu(x):
    return torch.max(x, torch.empty(x.size()))

# TODO
# def drelu(x):

def linear(x, w, b=None):
    """Applies a linear transformation to incoming data"""
    if x.dim() == 2 and b is not None:
        # fused op is marginally faster
        return torch.addmm(b, x, weight.t())
    else:
        output = x.matmul(weight.t())
        if b is not None:
            output += b
        return output
