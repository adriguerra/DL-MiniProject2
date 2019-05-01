import torch
import math

def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(1000)
