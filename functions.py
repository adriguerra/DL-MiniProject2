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

def linear(input, weights, bias=None):
    """Applies a linear transformation to incoming data"""
    if bias is not None:
        return torch.addmm(bias, input, weights.t())
    else:
        return torch.mm(input, weights.t())

# TODO
# def dlinear
