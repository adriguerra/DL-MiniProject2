import torch
import math
from helpers import generate_disc_set
from helpers import convert_to_one_hot_labels
from optimizer import SGD
import modules

torch.set_grad_enabled(False)

# Generate data
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
# Convert to one-hot labels
train_target = convert_to_one_hot_labels(train_input, train_target)
test_target = convert_to_one_hot_labels(test_input, test_target)
# Avoid vanishing gradient
train_input = 0.9 * train_input

# Define models
model1 = modules.Sequential(modules.Linear(2, 25),
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
model2 = modules.Sequential(modules.Linear(2, 25),
                            modules.ReLU(),
                            modules.Linear(25, 25),
                            modules.ReLU(),
                            modules.Linear(25, 25),
                            modules.ReLU(),
                            modules.Linear(25, 25),
                            modules.ReLU(),
                            modules.Linear(25, 2),
                            modules.MSELoss()
                           )
# Define training parameters
nb_epochs = 50
lr = 1e-5
mini_batch_size = 100
optimizer1 = SGD(model1.param(), lr=lr)
optimizer2 = SGD(model2.param(), lr=lr)


print("#" * 50)
print("Training model 1")
# Train model 1
for e in range(nb_epochs):
    sum_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        output, loss = model1.forward(train_input.narrow(0, b, mini_batch_size), train_target.narrow(0, b, mini_batch_size))
        grad = model1.backward()
        sum_loss = sum_loss + loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
    print("Iteration {0:}: loss = {1:.3f}".format(e+1, sum_loss), end='\r', flush=True)
loss_train = sum_loss

print()
print("#" * 50)

# Test model 1
output_test, loss_test = model1.forward(test_input, test_target)

# Print results
print("Model 1 with TanH results")
print("Training loss: {:.3f}".format(loss_train))
print("Test loss: {:.3f}".format(loss_test))


print("#" * 50)
print("Training model 2")

# Train model 2
for e in range(nb_epochs):
    sum_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        output, loss = model2.forward(train_input.narrow(0, b, mini_batch_size), train_target.narrow(0, b, mini_batch_size))
        grad = model2.backward()
        sum_loss = sum_loss + loss.item()
        optimizer2.step()
        optimizer2.zero_grad()
    print("Iteration {0:}: loss = {1:.3f}".format(e+1, sum_loss), end='\r', flush=True),

loss_train = sum_loss

print()
print("#" * 50)

# Test model 2
output_test, loss_test = model2.forward(test_input, test_target)
print("Model 2 with ReLu results")
print("Training loss: {:.3f}".format(loss_train))
print("Test loss: {:.3f}".format(loss_test))
