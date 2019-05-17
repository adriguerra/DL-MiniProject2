import torch
from helpers import generate_disc_set
from helpers import convert_to_one_hot_labels
from helpers import compute_nb_errors
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
# Normalize data
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

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
    nb_err1 = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        output, loss = model1.forward(train_input.narrow(0, b, mini_batch_size), train_target.narrow(0, b, mini_batch_size))
        optimizer1.zero_grad()
        grad = model1.backward()
        sum_loss = sum_loss + loss.item()
        optimizer1.step()
        nb_err1 = nb_err1 + compute_nb_errors(output, train_target.narrow(0, b, mini_batch_size))

    print("Iteration {0:}: loss = {1:.3f}".format(e+1, sum_loss), end='\r', flush=True)

loss_train1 = sum_loss/(train_input.shape[0]/mini_batch_size)

print()
print("#" * 50)

# Test model 1
output_test1, loss_test1 = model1.forward(test_input, test_target)
nb_err_test1 = compute_nb_errors(output_test1, test_target)

# Print results
print("Model 1 with TanH results")
print("Training loss: {:.3f}".format(loss_train1))
print("Test loss: {:.3f}".format(loss_test1))
print("Number of errors on train set: {:.2f}%".format((100*(nb_err1/train_target.shape[0]))))
print("Number of errors on test set: {:.2f}%".format((100*(nb_err_test1/train_target.shape[0]))))


print("#" * 50)
print("Training model 2")

# Train model 2
for e in range(nb_epochs):
    sum_loss = 0
    nb_err2 = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        output, loss = model2.forward(train_input.narrow(0, b, mini_batch_size), train_target.narrow(0, b, mini_batch_size))
        optimizer2.zero_grad()
        grad = model2.backward()
        sum_loss = sum_loss + loss.item()
        optimizer2.step()
        nb_err2 = nb_err2 + compute_nb_errors(output, train_target.narrow(0, b, mini_batch_size))

    print("Iteration {0:}: loss = {1:.3f}".format(e+1, sum_loss), end='\r', flush=True),

loss_train2 = sum_loss/(train_input.shape[0]/mini_batch_size)

print()
print("#" * 50)

# Test model 2
output_test2, loss_test2 = model2.forward(test_input, test_target)
nb_err_test2 = compute_nb_errors(output_test2, test_target)

print("Model 2 with ReLu results")
print("Training loss: {:.3f}".format(loss_train2))
print("Test loss: {:.3f}".format(loss_test2))

print("Number of errors on train set: {:.2f}%".format((100*(nb_err2/train_target.shape[0]))))
print("Number of errors on test set: {:.2f}%".format((100*(nb_err_test2/train_target.shape[0]))))
