from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# Add 2 tensors
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

# Provide an output tensor as an argument
result = torch.empty(5, 3)
torch.add(x,  y, out=result)
print(result)

# Addition - in place
y.add_(x)
print(y)

# You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing/reshaping
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# If you have a a one element tensor, use item() to get the value as a Python number
x = torch.rand(1)
print(x)
print(x.item())
