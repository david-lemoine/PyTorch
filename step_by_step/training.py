import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear layer with single 
        # input and single output
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
         # Now it only takes a call to the layer to make predictions
        return self.linear(x)

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

torch.manual_seed(97875)

# Now we can create a model and send it at once to the device
model = LayerLinearRegression().to(device)

# We can also inspect its parameters using its state_dict
print(model.state_dict())

lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

def make_train_step(model, loss_fn, optimizer):
    # Build function that performs one step in the train loop
    def train_step(x, y):
        # Set model to TRAIN mode
        model.train()

        # Make prediction
        yhat = model(x)

        # Compute loss
        loss = loss_fn(y, yhat)

        # Compute gradient
        loss.backward()

        # Update parameters and set the gradient to 0
        optimizer.step()
        optimizer.zero_grad()

        # Return the loss
        return loss.item()

    # Return the function that will be called inside the train loop
    return train_step

# Create the train_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)
losses = []

for epoch in range(n_epochs):
    # Perform one train step and return the corresponding loss
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

# Check model's parameters
print(model.state_dict())
