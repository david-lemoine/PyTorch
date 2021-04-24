# In this tutorial, we will stick with a simple and familiar problem: 
# a linear regression with a single feature x! It doesn’t get much simpler 
# than that…
#
#     Our model is y = a + bx + epsilon
#

# Lets us start by generating some synthetic data: we start with a vector 
# of 100 points for our feature x and create our labels using a = 1, b = 2 
# and some Gaussian noise.

import numpy as np 

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

# For this regression problem, we define the objective function, or loss, as
# the Mean Square Error (MSE), that is, the average of the squared differences
# between the labels (y) and the predictions (a + bx).

# We want to find (a, b) that minimize the loss function. We will do this using
# gradient descent. That is, at each point (a, b), we calculate the gradient of
# the loss function with respect to a and b.

# Since L = (1 / N) * \sum_{i = 1}^N (y_i - a - b x_i)^2, we have
#
# \frac{\partial L}{\partial a} = -(2 / N) \sum_{i = 1}^N (y_i - a - b x_i) 
#
# \frac{\partial L}{\partial b} = -(2 / N) \sum_{i = 1}^N x_i (y_i - a - b x_i) 

# Then at each iteration, we update the estimated values of a and b that minimize L
# by using gradient descent:
#
#   a <- a - \eta \frac{\partial L}{\partial a}
#   b <- b - \eta \frac{\partial L}{\partial b}
#
# where \eta is the learning rate.
#
# We then use the updated parameters to go back to step 1 (compute the loss) and repeat
# the process.
#
# An epoch is complete whenever every point has been already used for computing the loss. 
# For batch gradient descent, this is trivial, as it uses all points for computing the 
# loss — one epoch is the same as one update. For stochastic gradient descent, one epoch
# means N updates, while for mini-batch (of size n), one epoch has N/n updates.
# Repeating this process over and over, for many epochs, is, in a nutshell, training a model.

# For each epoch, there are four training steps:
# 
#  1. Compute the model’s predictions — this is the forward pass.
#  2. Compute the loss, using predictions and labels and the appropriate loss function 
#     for the task at hand.
#  3. Compute the gradients for every parameter.
#  4. Update the parameters.
#
# Just keep in mind that, if you don’t use batch gradient descent (our example does),
# you’ll have to write an inner loop to perform the four training steps for either each 
# individual point (stochastic) or n points (mini-batch). We’ll see a mini-batch example 
# later down the line.

# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

# Sets learning rate
lr = 1e-1

# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Compute the model predicted output
    yhat = a + b * x_train

    # Compute the error
    error = (y_train - yhat)
    loss = (error ** 2).mean()

    # Compute the gradient
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    # Update the parameters using the gradient and learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])
