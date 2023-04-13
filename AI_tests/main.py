import torch
import numpy as np
import pandas as pd

# The PyTorch autograd package provides automatic differentiation for all operations on Tensors.
# This is because operations become attributes of the tensors themselves.
# When a Tensor's .requires_grad attribute is set to True, it starts to track all operations on it.
# When an operation finishes you can call .backward() and have all the gradients computed automatically.
# The gradient for a tensor will be accumulated into its .grad attribute.

x= torch.tensor(2.0, requires_grad=True)
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
print(y)
print(type(y))
y.backward()            # activate backprop for y
print(x.grad)           # 93 is the result of the derivative of that function
                        # This is  the slope of the polynomial at the point (2, 63)

x=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True)      #set to False to disable tracking
print(x)
y=3*x+2
print(y)

z=2*y**2                # z is a second layer here, connected to the y equation
print(z)

out=z.mean()
print(out)

out.backward()          # perform backpropagation to find the gradient of X with respect to the output layer
print(x.grad)