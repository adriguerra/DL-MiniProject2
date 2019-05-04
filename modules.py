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
        self.input = input
        return linear(input, self.weight, self.bias)

    def backward(self, gradwrtoutput):
        self.gradwrtoutput = gradwrtoutput
        return linear(gradwrtoutput, self.weight, bias=None)

    def param(self):
        return [(self.input, self.gradwrtoutput)]

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
            setattr(self, module.__class__.__name__ + str(idx), module)

    def forward(self, input):
        """Apply forward pass sequentially on every module."""
        for module in self.__dict__.values():
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput):
        """Apply backward pass sequentially on every module."""
        # TODO First backward pass: gradwrtoutput = dloss(output, target)
        for module in self.__dict__.values().reverse():
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    def param(self):
        """Return a list of pairs, each composed of a parameter tensor,
        and a gradient tensor of same size."""
        params = []
        for module in self.__dict__.values():
            params.extend(module.param())
        return params

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return functions.loss(input, target)

    def backward(self, gradwrtoutput):
        raise NotImplementedError
