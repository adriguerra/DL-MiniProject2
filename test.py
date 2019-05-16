import torch
import math
from helpers import generate_disc_set
from helpers import convert_to_one_hot_labels
from optimizer import SGD
import modules

torch.set_grad_enabled(False)

# Generate data
train_input, train_target = generate_disc_set(1000)
# Convert to one-hot labels
train_target = convert_to_one_hot_labels(train_input, train_target)
# Example run
architecture = modules.Sequential(modules.Linear(2, 25),
                                  modules.TanH(),
                                  modules.Linear(25, 25),
                                  modules.TanH(),
                                  modules.Linear(25, 25),
                                  modules.TanH(),
                                  modules.Linear(25, 25),
                                  modules.TanH(),
                                  modules.Linear(25, 2),
                                  modules.MSELoss()
                                 )
