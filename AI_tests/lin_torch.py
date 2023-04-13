import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn  # import torch neural network functionality

X = torch.linspace(1, 50, 50).reshape(-1, 1)    # equally spaced matrix 1 by 50
print(X)

torch.manual_seed(71)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
print(e)

y = 2 * X + 1 + e   # slope is 2 and y-intercept is 1 with some noise
print(y.shape)



# we want to use pytorch and fit a simple regression line on the scatter plot
# y doesn't have a gradient function, so y.backward won't actually work

# how the actual built-in neutral network that linear model, preselects weight and bias value at random

torch.manual_seed(59)
model=nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)


class Model(nn.Module):
    def __init__(self, in_features, out_features): # when instantiated pass the size of incoming features and outgoing features
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  #describes the type of nn layer - linear layer - Dense

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

torch.manual_seed(59)
model = Model(1, 1)

print(model.linear.weight)      # initial layer weight and bias
print(model.linear.bias)

# Without seeing any data, the model sets a random weight of 0.1060 and a bias of 0.9638.

x = torch.tensor([2.0])
print(model.forward(x))     # equivalent to print(model(x))

# As models get more complex it may be better to iterate over all the model's parameters
# To find out what all the model's parameters are we can do the following:

for name, param in model.named_parameters():        # inherited from the nn.Module
    print(name, 't', param.item())

x=torch.tensor([2.0])
print(model.forward(x))

x1=np.linspace(0.0,50.0,50)
print(x1)

w1 = 0.1059
b1 = 0.9637

# if you don't set up any loss function or optimization at all, the model will
# essentially be guessing off its first random weight and random bias.
y1=w1*x1+b1
print(y1)
plt.scatter(X.detach().numpy(), y.detach().numpy())
plt.plot(x1,y1,"r")
#plt.show()

# Create the loss function

criterion=nn.MSELoss()      # this is how we label the loss function, based on this we value the performance
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)   # learning rate, inherited from nn.Module
epochs=50
losses=[]  #keep track of MSE

for i in range(epochs):
    i=i+1
    y_pred=model.forward(X)     # running through the data in a forward step X
    loss=criterion(y_pred,y)    # criterion between the y_pred values and the true y values, calculate our error
    losses.append(loss.detach().numpy())   # record that error. need to detach the tensor as it can't be called on a tensor that has a gradient function attached to it

    print(f"epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}")

    # gradients accumulate with every backpropagation - to prevent compounding we need to reset the stored gradient

    # we need to reset that sort of gradients for the new epoch
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    # keeping track of all the weights and biases of every neuron isn't important in a big network
    # more important to keep track of your loss and whether you are hitting convergence

plt.plot(range(epochs),losses)
plt.ylabel("MSE LOSS")
plt.xlabel("Epoch")
#plt.show()

# This error occurs because you are trying to call the numpy() method on a tensor that requires gradients.
# This is not allowed because the numpy() method creates a numpy array from the tensor, which breaks the connection
# to the computation graph and makes it impossible to calculate gradients.

x= np.linspace(0.0,50.0,50)
current_weight= model.linear.weight.item()
current_bias= model.linear.bias.item()

predicted_y=current_weight*x + current_bias

print(predicted_y)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, "r")
plt.show()

