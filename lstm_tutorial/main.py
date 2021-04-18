import bz2
from collections import Counter
import re
import nltk
import numpy as np

#nltk.download('punkt')

train_file = bz2.BZ2File('../data/amazon_reviews/train.ft.txt.bz2')
test_file = bz2.BZ2File('../data/amazon_reviews/test.ft.txt.bz2')

train_file = train_file.readlines()
test_file = test_file.readlines()

# This dataset contains a total of 4 million reviews - 3.6 million
# training and 0.4 million for testing. We will be using only 800k 
# for training and 200k for testing here -- this is still a large 
# amount of data.

# We're training on the first 800,000 reviews in the dataset
num_train = 800000  

# Using 200,000 reviews from test set
num_test = 200000

train_file = [x.decode('utf-8') for x in train_file[:num_train]]
test_file = [x.decode('utf-8') for x in test_file[:num_test]]

# We'll have to extract out the labels from the sentences. The data 
# is the format __label__1/2 <sentence>, therefore we can easily split
# it accordingly. Positive sentiment labels are stored as 1 and negative 
# are stored as 0.

# We will also change all URLs to a standard <url\> as the exact URL is 
# irrelevant to the sentiment in most cases.

# Extracting labels from sentences
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

# Some simple cleaning of data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
        
for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

# After quickly cleaning the data, we will do tokenization of the sentences, which is a 
# standard NLP task.

# There are many NLP libraries that can do this, such as spaCy or Scikit-learn, but 
# we will be using NLTK here as it has one of the faster tokenizers. The words will 
# then be stored in a dictionary mapping the word to its number of appearances. These 
# words will become our vocabulary.

# Dictionary that will map a word to the number of times it appeared in all the 
# training sentences
words = Counter()

for i, sentence in enumerate(train_sentences):
    # The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence):
        words.update([word.lower()])
        train_sentences[i].append(word)
    if i%20000 == 0:
        print(str((i*100)/num_train) + "% done")

print("100% done")

# To remove typos and words that likely don't exist, we'll remove all words from the 
# vocab that only appear once throughout

# Removing the words that only appear once
words = {k:v for k, v in words.items() if v > 1}

# Sorting the words according to the number of appearances, with the most common 
# word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD', '_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i, o in enumerate(words)}
idx2word = {i:o for i, o in enumerate(words)}

# With the mappings, we'll convert the words in the sentences to their corresponding indexes
for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# In the last pre-processing step, we'll be padding the sentences with 0s and 
# shortening the lengthy sentences so that the data can be trained in batches 
# to speed things up

# Defining a function that either shortens sentences or pads sentences with 0 
# to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype = int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# The length that the sentences will be padded/shortened to
seq_len = 200  

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Our dataset is already split into training and testing data. However, we still 
# need a set of data for validation during training. Therefore, we will split our 
# test data by half into a validation set and a testing set

# See also: https://machinelearningmastery.com/difference-test-validation-datasets/

split_frac = 0.5
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

# Next, this is the point where we’ll start working with the PyTorch library. We’ll 
# first define the datasets from the sentences and labels, followed by loading them 
# into a data loader. We set the batch size to 256. This can be tweaked according 
# to your needs

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

# See https://jamesmccaffrey.wordpress.com/2020/08/12/pytorch-dataset-and-dataloader-bulk-convert-to-tensors/

batch_size = 400

train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_data, shuffle = True, batch_size = batch_size)

# We can also check if we have any GPUs to speed up our training time by many folds. 
# If you’re using FloydHub with GPU to run this code, the training time will be 
# significantly reduced

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, 
# else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device 
# variable later in our code
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# At this point, we will be defining the architecture of the model. At this stage, 
# we can create Neural Networks that have deep layers or a large number of LSTM layers 
# stacked on top of each other. However, a simple model such as the one below with just
# an LSTM and a fully connected layer works quite well and requires much less training
# time. We will be training our own word embeddings in the first layer before the 
# sentences are fed into the LSTM layer.

# The final layer is a fully connected layer with a sigmoid function to classify 
# whether the review is of positive/negative sentiment

import sentiment_net as sn

# With this, we can instantiate our model after defining the arguments. The output 
# dimension will only be 1 as it only needs to output 1 or 0. The learning rate, loss 
# function and optimizer are defined as well

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = sn.SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

lr = 0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Finally, we can start training the model. For every 1000 steps, we’ll be checking 
# the output of our model against the validation dataset and saving the model if it 
# performed better than the previous time

epochs = 2
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter % print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# After we’re done training, it's time to test our model on a dataset it has 
# never seen before - our test dataset. We'll first load the model weights from 
# the point where the validation loss is the lowest.

# We can calculate the accuracy of the model to see how accurate our model’s 
# predictions are.

# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))

test_acc = num_correct / len(test_loader.dataset)

print("Test accuracy: {:.3f}%".format(test_acc*100))

