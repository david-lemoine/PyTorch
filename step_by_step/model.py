import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# In PyTorch, a model is represented by a regular Python class that inherits from the Module class.
#
# The most fundamental methods it needs to implement are:
#
#   __init__(self): it defines the parts that make up the model — in our case, two parameters, a and b.
# 
# You are not limited to defining parameters, though. Models can contain other models (or layers) as its 
# attributes as well, so you can easily nest them. We’ll see an example of this shortly as well.
#
#   forward(self, x): it performs the actual computation, that is, it outputs a prediction, given the input x.
#

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x

np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.rand(100, 1)

# Shuffle the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Next, we split our data into training and validation sets, using the first
# 80 shuffled points for training
train_idx = idx[:80]

# Use the remaining indices for validation
val_idx = idx[80:]

# Generate the train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Get a device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform the data to PyTorch tensors and then send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

torch.manual_seed(58375)

# Now we can create a model and send it at once to the device
model = ManualLinearRegression().to(device)

# We can also inspect its parameters using its state_dict
print(model.state_dict())

lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()

    # No more manual prediction!
    yhat = model(x_train_tensor)

    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())

# A note about the model.train() method... In PyTorch, models have a train() method 
# which, somewhat disappointingly, does NOT perform a training step. Its only purpose 
# is to set the model to training mode. Why is this important? Some models may use 
# mechanisms like Dropout, for instance, which have distinct behaviors in training 
# and evaluation phases.

