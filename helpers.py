import torch
import math

def generate_disc_set(nb):
    """Generates a data set (input and target) sampled uniformly between 0 and
    1 with a label 0 if outside the disk of radius 1/sqrt(2pi) and 1 inside."""
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return input, target

def convert_to_one_hot_labels(input, target):
    """Converts input to a one-hot encoded vector."""
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def compute_nb_errors(pred, target):
    """Computes the number of errors between the predicted output and the
    target."""
    nb_errors = 0
    _, predicted_classes = pred.max(1)
    _, target_classes = target.max(1)
    for k in range(target.shape[0]):
        if predicted_classes[k] != target_classes[k]:
            nb_errors = nb_errors + 1
    return nb_errors
