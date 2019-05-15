import torch
import math

def _calculate_fan_in_and_fan_out(tensor):
    """Input and output dimension of tensor."""
    fan_in = tensor.size(1)
    fan_out = tensor.size(0)
    return fan_in, fan_out

def _no_grad_normal_(tensor, mean, std):
    """Returns initialized tensor by given normal distribution parameters."""
    return tensor.normal_(mean, std)

def xavier_normal_(tensor):
    """Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std)
