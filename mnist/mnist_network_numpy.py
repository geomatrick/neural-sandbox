

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) # calculate the number of layers in the network
        self.sizes = sizes # pass sizes onto the class
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # create an array of column vectors of length y of random gaussian values for the biases of each layer except the input layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # create an array of (y by x) matrices of random gaussian values for the weights of each layer except the input layer. The dimensions of the weight matrix for layer l are (number of neurons in layer l, number of neurons in layer l-1)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): # loop through the tuple of weight and biases, which is possible because each is an array of length num_layers - 1, and the weights and biases are ordered by layer
            a = sigmoid(np.dot(w, a) + b) # this is the formula for calculating the output of a layer, where w is the weight matrix for the layer, a is the input to the layer, and b is the bias vector for the layer. The dot product of w and a gives us the weighted input to each neuron in the layer, and then we add the bias to get the total input to each neuron. Finally, we apply the sigmoid function to get the output of each neuron in the layer.
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # loop through the number of epochs
        # for each epoch, we random select mini batches from the training data, until we have exhausted the training data.
        # For each mini batch, we run feedforward to caluclate the output of the network, then calculate the error - or loss - of the output, and then use backpropogation to calculate the gradient of the cost function with respect to the weights and biases, and then update the weights and biases using gradient descent. After each epoch, if test data is provided, we evaluate the network on the test data and print out the progress.
        for j in range(epochs):
            random.shuffle(training_data) # shuffle the training data so that consecutive slices form effectively random mini batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] # partition the training data into mini batches of size mini_batch_size
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch. The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate.
            if test_data: print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)) # for test data, it prints the epoch number, the number of test inputs for which the neural network outputs the correct result, and the total number of test inputs. Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
            else: print("Epoch {0} complete".format(j)) # if not test data, it just indicates that the epoch has finished

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] # nabla is the upside down triangle symbol used for the gradient. So this is the gradient with respect to the biases, initialised to an array of zeros with the same shape as the biases for each layer
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Same as above but with respect to the weights
        for x, y in mini_batch: # Loop through the inputs and the desired outputs in the mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # calculate the gradient of the cost function with respect to the weights and biases for a single input and desired output using backpropagation. This returns two lists of numpy arrays, one for the gradient with respect to the biases and one for the gradient with respect to the weights, where each list is ordered by layer.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # add the nabla_b outputs of the backprop function to the existing nabla_b array
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # Same as above but for weights
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)] # Update the biases by subtracting the learning rate time the average gradient for the mini batch from the existing biases. The average gradient is calculated by dividing the total gradient by the size of the mini batch, which is len(mini_batch).
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)] # Same as above but for weights

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] # nabla is the upside down triangle symbol used for the gradient. So this is the gradient with respect to the biases, initialised to an array of zeros with the same shape as the biases for each layer
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Same as above but with respect to the weights
        # feedforward (but we can't use the above feedforward function because we want to store all the intermediate z vectors and activations)
        activation = x # the input layer activation is just the input x
        activations =[x] # list to store all the activations, layer by layer
        z_array = [] # list to store all the z vectors, layer by layer
        for (b, w) in zip(self.biases, self.weights): # loop through the tuple of weights and biases
            z = np.dot(w, activation) + b # calculate the z vector for the layer, which is the weighted input to the neurons in the layer. This is calculated by taking the dot product of the weight matrix and the activation from the previous layer, and then adding the bias vector for the layer.
            z_array.append(z) # add the z vector to the list of z vectors
            activation = sigmoid(z) # Calculate the activation for the layer by applying the sigmoid function to the z value, and then update the activation variable to be used as the input for the next layer
            activations.append(activation) # Add the activation for the layer to the list of activations
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(z_array[-1]) # This is the first equation of back propogation from Michael Nielsen's book, which calculates the error delta for the output layer. The cost_derivative function calculates the derivative of the cost function with respect to the output activations, and then we multiply that by the derivative of the sigmoid function applied to the z vector for the output layer to get the error delta for the output layer.
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # This is the fourth equation of back propagation (BP4) from Michael Nielsen's book, which calculates the gradient of the cost function with respect to the weights for the output layer. This is calculated by taking the dot product of the error delta for the output layer and the transpose of the activations from the previous layer (which is the second-last layer). The result is a matrix where each element (j, k) is the gradient of the cost function with respect to the weight connecting neuron k in the second-last layer to neuron j in the output layer.
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): # loop through the layers in reverse
            z = z_array[-l] # get the z vector for the layer that we saved earlier into our z array
            sp = sigmoid_prime(z) # we need this to keep calculating the second equation of back propogation for the hidden layers, which requires us to calculate the derivative of the sigmoid function applied to the z vector for the layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # More of the second equation
            nabla_b[-l] = delta # the gradient with respect to the biases for the layer is just the error delta for the layer
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # the gradient with respect to the weights for the layer is calculated by taking the dot product of the error delta for the layer and the transpose of the activations from the previous layer (which is the layer before the one we are currently calculating the gradient for)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] # Argmax returns the index of the neuron in the output with the highest activation
        return sum(int(x == y) for (x, y) in test_results) # Return the number of correct results


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))