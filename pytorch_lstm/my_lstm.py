import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        # LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True)
        # First fully connected layer
        self.fc_1 =  nn.Linear(hidden_size, 128)
        # Last fully connected layer
        self.fc = nn.Linear(128, num_classes)
        # ReLU activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Internal state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        # Reshape the data for Dense layer next
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        # First dense layer
        out = self.fc_1(out)
        # Relu
        out = self.relu(out)
        # Final output
        out = self.fc(out)
        return out