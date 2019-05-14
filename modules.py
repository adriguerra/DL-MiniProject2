import torch
import functions
import math
import init

class Module(object):

    def forward(self, input):
        """Apply the module's forward pass and save the input as a parameter."""
        raise NotImplementedError

    def backward(self, gradswrtoutput):
        """Apply the module's backward pass and save the gradient with respect
        to output as a parameter."""
        raise NotImplementedError

    def param(self):
        """Return a list of pairs, each composed of a parameter tensor,
        and a gradient tensor of same size. Typically passed to optimizer."""
        return []

class Linear(Module):
    """Applies a linear transformation to incoming data."""
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.zeros(out_features, in_features)
        self.dweight = torch.zeros(out_features, in_features)

        if bias:
            self.bias = torch.zeros(out_features)
            self.dbias = torch.zeros(out_features)
        else:
            self.bias = None
            self.dbias = None

        self.reset_parameters()
        super().__init__()

    def forward(self, input):
        """Applies the forward pass by returning a linear function wx + b,
        where x is the input, w the weights and b the bias and saves the input
        for the backward pass to compute its computations."""
        self.input = input
        return functions.linear(self.input, self.weight, self.bias)

    def backward(self, gradwrtoutput):
        """Applies the backward pass by computing the loss derived with respect
        to the input and saves the derivatives of the loss with respect to the
        weights and bias."""
        # Save the gradient with respect to the output
        self.gradwrtoutput = gradwrtoutput
        # Derivative of the loss with respect to weight
        self.dweight = gradwrtoutput.t().mm(self.input)
        # Derivative of the loss with respect to bias
        self.dbias = self.gradwrtoutput.t().sum(1)
        return gradwrtoutput.mm(self.weight)

    def param(self):
        """Returns the module's weights, weight derviatives, bias and bias
        derivative in a list of tuples."""
        if self.bias is None and self.dbias is None:
            return [(self.weight, self.dweight)]
        else:
            return [(self.weight, self.dweight), (self.bias, self.dbias)]

    def reset_parameters(self):
        """Initialize weights using normal distribution"""
        epsilon = 1e-6
        self.weight = init.xavier_normal_(self.weight)
        if self.bias is not None:
            self.bias = self.bias.normal_(0, epsilon)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """Apply the ReLu activation function on the input and save the input."""
        self.input = input
        return functions.relu(input)

    def backward(self, gradwrtoutput):
        """Returns the loss derived with respect to the output"""
        return gradwrtoutput * functions.drelu(self.input)

    def param(self):
        return []

class TanH(Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, input):
        """Apply the hyperbolic tangent activation function on the input and
        save the input."""
        self.input = input
        return functions.tanh(input)

    def backward(self, gradwrtoutput):
        """Returns the loss derived with respect to the output"""
        return gradwrtoutput * functions.dtanh(self.input)

    def param(self):
        return []

class Sequential(Module):
    def __init__(self, *args):
        if not isinstance(args[-1], MSELoss):
            raise TypeError("Last module must be a loss.")

        for idx, module in enumerate(args):
            module_name = module.__class__.__name__
            setattr(self, module_name + str(idx), module)
        super().__init__()

    def forward(self, input):
        """Apply forward pass sequentially on every module."""
        for module in self.__dict__.values():
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput=1):
        """Apply backward pass sequentially on every module."""
        for module in list(self.__dict__.values())[::-1]:
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    def param(self):
        """Return a list of pairs, each composed of a parameter tensor,
        and a gradient tensor of same size. Typically passed to optimizer."""
        params = []
        for module in self.__dict__.values():
            params.extend(module.param())
        return params

class MSELoss(Module):
    def __init__(self, target):
        self.target = target.float()
        super().__init__()

    def forward(self, output):
        """Compute the mean-squared error of the output and the target."""
        self.output = output.float()
        loss = functions.loss(self.output, self.target)
        return (output, loss)

    def backward(self, gradswrtoutput=1):
        """Compute the derivative of the mean-squared error of the output and
        the target."""
        dloss = gradswrtoutput * functions.dloss(self.output, self.target)
        return dloss
