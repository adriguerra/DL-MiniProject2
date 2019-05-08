import torch
import math

def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return input, target

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

train_input, train_target = generate_disc_set(1000)
train_target = convert_to_one_hot_labels(train_input, train_target)
