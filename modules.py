import torch
import functions

class Module(object):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradswrtoutput):
        raise NotImplementedError
    def param(self):
        return []

class Linear(Module):
    """Applies a linear transformation to incoming data"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features)
        if bias:
            self.bias = torch.Tensor(out_features)
        else:
            self.bias = None

    def forward(self, input):
        s = linear(input, self.weight, self.bias)
        return s

    def backward(self, gradwrtoutput):
        return linear(gradwrtoutput, self.weight, bias=None)

    def param(self):
        return self.weight, self.bias

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        return functions.relu(input)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * functions.drelu(self.input)

    def param(self):
        return []

class TanH(Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, input):
        return functions.tanh(input)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * functions.dtanh(self.input)

    def param(self):
        return []

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            setattr(self, idx, module)

    def forward(self, input):
        """Apply forward sequentially on every module"""
        for attr in dir(self):
            module = getattr(self, attr)
            input = module(input)
        return input

    def backward(self, input):
        return functions.relu(input)

    def param(self):
        return []
