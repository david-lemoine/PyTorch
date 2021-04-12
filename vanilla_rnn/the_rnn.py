import torch
import torch.nn as nn

# To start building our own neural network model, we can define a class that 
# inherits PyTorch’s base class (nn.module) for all neural network modules. After 
# doing so, we can start defining some variables and also the layers for our model 
# under the constructor. For this model, we’ll only be using 1 layer of RNN followed 
# by a fully connected layer. The fully connected layer will be in charge of converting 
# the RNN output to our desired output shape.

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers...
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden
        
    # This method generates the first hidden state of zeros which we'll use in the forward pass
    # We'll send the tensor holding the hidden state to the device we specified earlier as well
    def init_hidden(self, batch_size): 
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
