import torch
import torch.nn as nn

# Represents the size of the input at each time step
input_dim = 5

# Represent the size of the hidden state and cell state at each time step
hidden_dim = 10

# The number of LSTM layers stacked on top of each other
n_layers = 1

lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

# Let's create some dummy data to see how the layer takes in the input.
# As our input dimension is 5, we have to create a tensor of the shape (1, 1, 5)
# which represents (batch size, sequence length, input dimension).

batch_size = 1
seq_len = 1

inp = torch.randn(batch_size, seq_len, input_dim)

# Additionally, we'll have to initialize a hidden state and cell state for 
# the LSTM as this is the first cell. The hidden state and cell state is 
# stored in a tuple with the format (hidden_state, cell_state).

hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

# Next, we’ll feed the input and hidden states and see what we’ll get back from it

out, hidden = lstm_layer(inp, hidden)
print("Output shape: ", out.shape)
print("Hidden: ", hidden)

# In the process above, we saw how the LSTM cell will process the input and 
# hidden states at each time step. However in most cases, we'll be processing 
# the input data in large sequences. The LSTM can also take in sequences of 
# variable length and produce an output at each time step. Let's try changing 
# the sequence length this time.

seq_len = 3
inp = torch.randn(batch_size, seq_len, input_dim)
out, hidden = lstm_layer(inp, hidden)
print(out.shape)

# This time, the output's 2nd dimension is 3, indicating that there were 3 outputs 
# given by the LSTM. This corresponds to the length of our input sequence. For the 
# use cases where we'll need an output at every time step (many-to-many), such as 
# Text Generation, the output of each time step can be extracted directly from the
# 2nd dimension and fed into a fully connected layer. For text classification tasks
# (many-to-one), such as Sentiment Analysis, the last output can be taken to be fed 
# into a classifier.

# Obtaining the last output
out = out.squeeze()[-1, :]
print(out.shape)