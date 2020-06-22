import random
import mnist_reader
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialize_biases(sizes)
        self.initialize_weights(sizes)

    # Initialize the bias vectors, for each layer l=1 to L.
    def initialize_biases(self, sizes):
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]

    # Initialize the weight matrices, for each layer l = 1 to L.
    def initialize_weights(self, sizes):
        self.weights = [np.random.randn(i, j) for i, j in zip(sizes[1:], sizes[:-1])]

    # Calculates the neural network output for the given inputs.
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a) + b)
        return a

    # Train the neural network using mini-batch stochastic gradient descent.
    # The "training_data" is a list of tuples "(x, y)" representing the training inputs
    # and the desired outputs.  The other non-optional parameters are self-explanatory.  
    # If "test_data" is provided then the network will be evaluated against the test data 
    # after each epoch, and partial progress printed out.  This is useful for tracking 
    # progress, but slows things down substantially.
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if (test_data): n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # Update the network's weights and biases by applying gradient descent using 
    # backpropagation to a single mini batch. The "mini_batch" is a list of tuples "(x, y)", 
    # and "eta" is the learning rate.
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # Calculates the gradient of the loss function with respect to the weights
    # and biases using backpropagation.
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Forward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        # Compute the error for the output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # Compute the gradient of the cost function with respect to the weights and biases
        # for the output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Backpropagate the error and compute the gradients for each layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.matmul(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # Return the number of test inputs for which the neural network outputs the 
    # correct result. Note that the neural network's output is assumed to be the 
    # index of whichever neuron in the final layer has the highest activation.
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    # Returns the gradient of the cost function with respect to the activations a, 
    # evaluated at the output layer.
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

# Sigmoid activation function.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# The derivative of the sigmoid activation function.
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# One-hot encoding of the labels.
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# This loads the training and test data.
def load_data_wrapper():
    x_train, y_train = mnist_reader.load_mnist('./data/fashion', kind='train')
    x_train_v = [np.reshape(x, (784, 1)) for x in x_train]
    y_train_v = [vectorized_result(y) for y in y_train]
    train_dataset = list(zip(x_train_v, y_train_v))
    x_test, y_test = mnist_reader.load_mnist('./data/fashion', kind='t10k')
    x_test_v = [np.reshape(x, (784, 1)) for x in x_test]
    #y_test_v = [vectorized_result(y) for y in y_test]
    test_dataset = list(zip(x_test_v, y_test))
    return train_dataset, test_dataset

# Load the training and test data
training_data, test_data = load_data_wrapper()

# Setup a network with 1 hidden layer with 64 neurons
net = Network([784, 32, 10])

net.SGD(training_data, 30, 10, 0.02, test_data=test_data)


