import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

# So far, we’ve simply used our Numpy arrays turned PyTorch tensors. 
# But we can do better, we can build a Dataset!

# In PyTorch, a dataset is represented by a regular Python class that 
# inherits from the Dataset class. You can think of it as a kind of a Python
# list of tuples, each tuple corresponding to one point (features, label).

# Let’s build a simple custom dataset that takes two tensors as arguments: 
# one for the features, one for the labels. For any given index, our dataset 
# class will return the corresponding slice of each of those tensors. It should 
# look like this:

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

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
    
# Wait, is this a CPU tensor now? Why? Where is .to(device)?
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

# We are building a Dataset because we want to use a DataLoader. Until now, 
# we have used the whole training data at every training step. It has been 
# batch gradient descent all along. This is fine for our small dataset, but 
# if we want to go serious about all this, we must use mini-batch gradient 
# descent. Thus, we need mini-batches. Thus, we need to slice our dataset accordingly.

# So we use PyTorch’s DataLoader class for this job. We tell it which dataset to use 
# (the one we just built in the previous section), the desired mini-batch size and 
# if we’d like to shuffle it or not.

# Our loader will behave like an iterator, so we can loop over it and fetch a
# different mini-batch every time.

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

torch.manual_seed(97875)

# Now we can create a model and send it at once to the device
model = LayerLinearRegression().to(device)

# We can also inspect its parameters using its state_dict
print(model.state_dict())

lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []
train_step = make_train_step(model, loss_fn, optimizer)

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = train_step(x_batch, y_batch)
        losses.append(loss)

print(model.state_dict())

