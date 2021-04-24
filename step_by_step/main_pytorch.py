import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# from torchviz import make_dot

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform the data to PyTorch tensors and then send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())

# What distinguishes a tensor used for data — like the ones we’ve just created — 
# from a tensor used as a (trainable) parameter/weight?
# The latter tensors require the computation of its gradients, so we can update 
# their values (the parameters’ values, that is). That’s what the 
# requires_grad=True argument is good for. It tells PyTorch we want it to 
# compute gradients for us.

# FIRST
# Initializes parameters "a" and "b" randomly, ALMOST as we did in Numpy
# since we want to apply gradient descent on these parameters, we need
# to set REQUIRES_GRAD = TRUE
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
print(a, b)

# SECOND
# But what if we want to run it on a GPU? We could just send them to device, right?
a = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
print(a, b)
# Sorry, but NO! The to(device) "shadows" the gradient...

# THIRD
# We can either create regular tensors and send them to the device (as we did 
# with our data)
a = torch.randn(1, dtype=torch.float).to(device)
b = torch.randn(1, dtype=torch.float).to(device)
# and THEN set them as requiring gradients...
a.requires_grad_()
b.requires_grad_()
print(a, b)

# In PyTorch, every method that ends with an underscore (_) makes changes in-place,
# meaning, they will modify the underlying variable.

# Although the last approach worked fine, it is much better to assign tensors to a 
# device at the moment of their creation.

# We can specify the device at the moment of creation - RECOMMENDED!
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

# Autograd is PyTorch’s automatic differentiation package. Thanks to it, we don’t 
# need to worry about partial derivatives, chain rule or anything like it.
# So, how do we tell PyTorch to do its thing and compute all gradients? That’s what
# backward() is good for.

# Do you remember the starting point for computing the gradients? It was the loss, 
# as we computed its partial derivatives w.r.t. our parameters. Hence, we need to 
# invoke the backward() method from the corresponding Python variable, like, 
# loss.backward().

# What about the actual values of the gradients? We can inspect them by looking at 
# the grad attribute of a tensor.

# If you check the method’s documentation, it clearly states that gradients are 
# accumulated. So, every time we use the gradients to update the parameters, we 
# need to zero the gradients afterwards. And that’s what zero_() is good for.

# That’s it? Well, pretty much… but, there is always a catch, and this time it has to 
# do with the update of the parameters...

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # No more manual computation of gradients! 
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    # Let's check the computed gradients...
    print(a.grad)
    print(b.grad)

    # What about UPDATING the parameters? Not so fast...
    
    # FIRST ATTEMPT
    # AttributeError: 'NoneType' object has no attribute 'zero_'
    # a = a - lr * a.grad
    # b = b - lr * b.grad
    # print(a)

    # SECOND ATTEMPT
    # RuntimeError: a leaf Variable that requires grad has been used in an in-place 
    # operation
    # a -= lr * a.grad
    # b -= lr * b.grad 

    # THIRD ATTEMPT
    # We need to use NO_GRAD to keep the update out of the gradient computation
    # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()
    
print(a, b)

# So far, we’ve been manually updating the parameters using the computed gradients. 
# That’s probably fine for two parameters… but what if we had a whole lot of them?! 
# We use one of PyTorch’s optimizers, like SGD or Adam.

# An optimizer takes the parameters we want to update, the learning rate we want to 
# use (and possibly many other hyper-parameters as well!) and performs the updates 
# through its step() method.

# Besides, we also don’t need to zero the gradients one by one anymore. We just invoke 
# the optimizer’s zero_grad() method and that’s it!

# In the code below, we create a Stochastic Gradient Descent (SGD) optimizer to update 
# our parameters a and b.

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

lr = 1e-1
n_epochs = 1000

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    loss.backward()

    # No more manual updates!
    optimizer.step()

    # No more telling PyTorch to let gradients go!
    optimizer.zero_grad()

print(a, b)

# We now tackle the loss computation. As expected, PyTorch got us covered 
# once again. There are many loss functions to choose from, depending on the
#  task at hand. Since ours is a regression, we are using the Mean Square 
# Error (MSE) loss.

torch.manual_seed(684865)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

lr = 1e-1
n_epochs = 1000

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    
    # No more manual loss!
    loss = loss_fn(y_train_tensor, yhat)

    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()
    
print(a, b)