from __future__ import print_function
import numpy as np
import torch

# Converting a Torch tensor to a NumPy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting a NumPy array to a Torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
