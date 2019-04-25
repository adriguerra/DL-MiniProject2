import torch
import functions

class Linear(torch.nn.Module):
"""Applies a linear transformation to incoming data"""

    def __init__(self, in_features, out_features, bias=True):
            super(Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.Tensor(out_features, in_features)
            if bias:
                self.bias = torch.Tensor(out_features)
            else:
                self.bias = None
            # self.reset_parameters()
    def forward(self, *input):
        return linear(input, self.weight, self.bias)

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError

class Sequential(torch.nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
