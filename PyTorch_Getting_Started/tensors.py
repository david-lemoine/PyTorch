from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled with zeros and of dtype long
x = torch.zeros(5, 3, dtype = torch.long)
print(x)

# Construct a tensor directly from data
#x = torch.tensor([5.5, 3])
#print(x)

# Create a tensor based on an existing tensor. These methods will 
# reuse properties of the input tensor, e.g. dtype, unless new 
# values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)

print(x.size())
