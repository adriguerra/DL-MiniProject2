# EE-559 Deep Learning: Mini-Project 2

The objective of this project is to design a mini “deep learning framework” using only PyTorch’s
tensor operations and the standard math library. In particular, without using autograd or the
neural-network modules. This project is part of *EE-559 Deep Learning* course taught  at  EPFL  during  the  spring  semester  2019.

The framework provides the necessary tools to:
 - build networks combining fully connected layers, Tanh, and ReLU
 - run the forward and backward passes
 - optimize parameters with SGD for MSE.

## Example run of our framework

```Python
# Assuming train_input, train_target, test_input and test_target given

# Imports
import modules
from optimizer import SGD

# Define architecture
model = modules.Sequential(
    modules.Linear(2, 25), 
    modules.TanH(), 
    modules.Linear(25, 2), 
    modules.MSELoss())

# Define parameters
nb_epochs = 50
lr = 1e-3
optimizer = SGD(model.param(), lr)

# Train model
for e in range(nb_epochs):
    output, loss = model.forward(train_input, train_target)
    optimizer.zero_grad()
    grad = model.backward()
    optimizer.step()

# Test model
output, loss_test = model.forward(test_input, test_loss)
```
