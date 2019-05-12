import torch
import math

def _calculate_fan_in_and_fan_out(tensor):
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
        return fan_in, fan_out

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)

def xavier_normal_(tensor):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std)
