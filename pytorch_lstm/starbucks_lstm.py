import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SBUX.csv", index_col = "Date", parse_dates=True)

df.head(5)

plt.style.use("ggplot")
df["Volume"].plot(label="CLOSE", title="Star Bucks Stock Volume")

# Get the data and the labels separate from a single dataframe
X = df.iloc[:, :-1]
y = df.iloc[:, 5:6]

# Use the standard scaler for the features and the min/max scaler
# for the output values
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 

# The next thing is splitting the dataset into 2 parts. 1 is for the 
# training, and the other part is for testing the values. Since it is 
# sequential data, and order is important,  you will take the first 200
# rows for training, and 53 for testing the data.
X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_mm[:200, :]
y_test = y_mm[200:, :] 

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 

# You can simply convert the Numpy Arrays to Tensors and to Variables 
# (which can be differentiated) via this simple code.
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

# Now the next step is to check the input format of an LSTM. This means 
# that since LSTM is specially built for sequential data, it can not take 
# in simple 2-D data as input. They need to have the timestamp information 
# with them too, as we discussed that we need to have input at each timestamp.
# So let’s convert the dataset.

# Reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

import my_lstm

num_epochs = 1000
learning_rate = 0.001

input_size = 5
hidden_size = 2
num_layers = 1

num_classes = 1

# Instantiate our lstm network
lstm1 = my_lstm.LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])

# Define the Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

# Now loop for the number of epochs, do the forward pass, calculate the loss, 
# improve the weights via the optimizer step:

for epoch in range(num_epochs):
    # Forward pass
    outputs = lstm1.forward(X_train_tensors_final)
    # Calculate the gradient, manually setting to 0
    optimizer.zero_grad()
    
    # Obtain the loss function
    loss = criterion(outputs, y_train_tensors)
    
    # Calculate the loss of the loss function
    loss.backward() 
    
    # Improve from loss, i.e backprop
    optimizer.step()
    
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# You can see that there is less loss, which means it is performing well. 
# Let’s plot the predictions on the data set, to check out how it’s performing.
# But before performing predictions on the whole dataset, you’ll need to bring 
# the original dataset into the model suitable format, which can be done by 
# using similar code as above.

df_X_ss = ss.transform(df.iloc[:, :-1])
df_y_mm = mm.transform(df.iloc[:, -1:])

# Convert to tensors
df_X_ss = Variable(torch.Tensor(df_X_ss)) 
df_y_mm = Variable(torch.Tensor(df_y_mm))

# Reshape the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

# We will now simply perform predictions on the whole dataset via a forward pass, 
# and then to plot them, we will convert the predictions to numpy, reverse transform
#  them (remember that we transformed the labels to check the actual answer, and that 
# we’ll need to reverse transform it) and then plot it.

# Forward pass
train_predict = lstm1(df_X_ss)

# Numpy conversion
data_predict = train_predict.data.numpy()
dataY_plot = df_y_mm.data.numpy()

# Reverse transformation
data_predict = mm.inverse_transform(data_predict)
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6))
plt.axvline(x=200, c='r', linestyle='--')

# Plot
plt.plot(dataY_plot, label='Actuall Data')
plt.plot(data_predict, label='Predicted Data')
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 