# This is based on this article:
#  https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import torch
from torch import nn
import numpy as np
import the_rnn

# First, we'll define the sentences that we want our model to output when 
# fed with the first word or the first few characters.

# Then we'll create a dictionary out of all the characters that we have in
#  the sentences and map them to an integer. This will allow us to convert 
# our input characters to their respective integers and vice versa.

text = ["hey how are you", "good i am fine", "have a nice day"]

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set("".join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# Next, we'll be padding our input sentences to ensure that all the sentences are of 
# standard length. While RNNs are typically able to take in variably sized inputs, we 
# will usually want to feed training data in batches to speed up the training process. 
# In order to used batches to train on our data, we'll need to ensure that each sequence
#  within the input data is of equal size.

# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))

# Padding

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until 
# the length of the sentence matches the length of the longest sentence
for i in range(len(text)):
  while len(text[i]) < maxlen:
      text[i] += ' '

# As we're going to predict the next character in the sequence at each time step, 
# we'll have to divide each sentence into input data and ground truth

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

# Remove last character for input sequence, and remove first character for target sequence
for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

# Now we can convert our input and target sequences to sequences of integers instead 
# of a sequence of characters by mapping them using the dictionaries we created above. 
# This will allow us to one-hot-encode our input sequence subsequently.

for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

# Before encoding our input sequence into one-hot vectors, we'll define 3 key variables:
#
# - dict_size: Dictionary size - The number of unique characters that we have in our text.
#   This will determine the one-hot vector size as each character will have an assigned 
#   index in that vector
#
# - seq_len: The length of the sequences that we're feeding into the model. As we standardized 
#   the length of all our sentences to be equal to the longest sentences, this value will be
#   the max length - 1 as we removed the last character input as well
#
# - batch_size: The number of sentences that we defined and are going to feed into the model 
#   as a batch

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

# One-hot encoding:
def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

# Since we're done with all the data pre-processing, we can now move the data from NumPy 
# arrays to PyTorch's very own data structure - Torch Tensors

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

# Before we start building the model, let's use a built-in feature in PyTorch to 
# check the device we're running on (CPU or GPU)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, 
# else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# We'll have to instantiate the model with the relevant parameters and define our 
# hyper-parameters as well. The hyper-parameters we're defining below are:
#
#  - n_epochs: Number of Epochs --> Number of times our model will go through the 
#    entire training dataset

#  - lr: Learning Rate --> Rate at which our model updates the weights in the cells 
#    each time back-propagation is done

# Similar to other neural networks, we have to define the optimizer and loss function 
# as well. We’ll be using CrossEntropyLoss as the final output is basically a 
# classification task and the common Adam optimizer.

# Instantiate the model with hyperparameters
model = the_rnn.Model(input_size = dict_size, output_size = dict_size, hidden_dim = 12, n_layers = 1)

# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr = 0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Now we can begin our training! As we only have a few sentences, this training process 
# is very fast.

# Training Run
for epoch in range(1, n_epochs + 1):
    # Clears existing gradients from previous epoch
    optimizer.zero_grad() 
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())

    # Do backpropagation and calculates gradients
    loss.backward()

    # Updates the weights accordingly
    optimizer.step() 
    
    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# This function takes in the model and character as arguments and returns the next 
# character prediction and hidden state.
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim = 0).data

    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim = 0)[1].item()

    return int2char[char_ind], hidden

# This function takes the desired output length and input characters as arguments, 
# returning the produced sentence.
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)

    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

# Let's run the function with our model and the starting words 'good'
sample(model, 15, 'good')
