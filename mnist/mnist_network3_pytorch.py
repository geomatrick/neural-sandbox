#### Libraries
import random
import torch
import torch.nn as nn
import torch.optim as optim

"""Rewrite of Nielsen's network3.py (originally in Theano) using PyTorch.

network3 introduces three things not in network 1 and 2:
  1. Convolutional + pooling layers (ConvPoolLayer)
  2. Dropout regularisation (FullyConnectedLayer)
  3. Softmax output with cross-entropy loss (SoftmaxLayer)

The Network class composes these layers in a list, so you can experiment
with different architectures by changing what you pass in.
"""


class ConvPoolLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pool_size,
                 activation=torch.relu):
        """A convolutional layer followed by max pooling.

        Unlike the fully-connected layers in network 1 and 2, conv layers share
        weights across spatial positions — one filter scans the whole image.

        ``in_channels``  — number of input feature maps (1 for greyscale MNIST)
        ``out_channels`` — number of filters (i.e. output feature maps)
        ``kernel_size``  — size of each filter (e.g. 5 means a 5x5 filter)
        ``pool_size``    — size of max pooling window (e.g. 2 means 2x2)

        Output spatial size after this layer for a 28x28 MNIST input:
          after conv:  (28 - kernel_size + 1) x (28 - kernel_size + 1)
          after pool:  divide each dimension by pool_size
        e.g. kernel_size=5, pool_size=2 → 12x12 feature maps."""
        super().__init__()
        self.activation = activation
        # define self.conv as nn.Conv2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # define self.pool as nn.MaxPool2d
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        """Input x has shape (batch, in_channels, height, width)."""
        return self.pool(self.activation(self.conv(x)))


class FullyConnectedLayer(nn.Module):

    def __init__(self, n_in, n_out, activation=torch.relu, dropout_rate=0.0):
        """A fully-connected layer with optional dropout.

        Same idea as network 1 and 2 from Nielsen but:
          - ReLU activation by default instead of sigmoid
          - Dropout randomly zeroes activations during training to reduce
            overfitting. Use nn.Dropout(dropout_rate).

        Dropout is automatically disabled when you call model.eval(),
        so you don't need to handle it manually during evaluation.

        ``n_in``         — number of input features
        ``n_out``        — number of output features
        ``activation``   — activation function (default: torch.relu)
        ``dropout_rate`` — fraction of activations to zero (0 = no dropout)"""
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


class SoftmaxLayer(nn.Module):

    def __init__(self, n_in, n_out, dropout_rate=0.0):
        """The output layer — a linear map followed by softmax.

        Replaces the sigmoid output from network 1 and 2. Softmax produces a
        probability distribution over the 10 digit classes.

        Do NOT apply softmax manually here. Use nn.CrossEntropyLoss in the
        training loop — it combines LogSoftmax + NLLLoss internally and is
        numerically more stable.

        For evaluation, use torch.argmax on the raw output (logits).

        ``n_in``         — number of input features
        ``n_out``        — number of classes (10 for MNIST)
        ``dropout_rate`` — applied before the linear layer"""
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Apply dropout → linear. CrossEntropyLoss handles the softmax."""
        return self.linear(self.dropout(x))


class Network(nn.Module):

    def __init__(self, layers):
        """Compose a network from a list of layer objects.

        Unlike network 1 and 2 where the architecture was fixed, here you pass
        in a list of already-constructed layer objects, e.g.:

            net = Network([
                ConvPoolLayer(1, 20, 5, 2),
                FullyConnectedLayer(20*12*12, 100, dropout_rate=0.5),
                SoftmaxLayer(100, 10)
            ])

        Store the layers in a nn.ModuleList so PyTorch registers them."""
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Pass x through each layer in sequence.

        Two shape issues to handle:
          1. Input from mnist_loader is shape (batch, 784) — a flat vector.
             If the first layer is a ConvPoolLayer, reshape to (batch, 1, 28, 28)
             before passing in

          2. Between a ConvPoolLayer and a FullyConnectedLayer, the tensor
             needs to be flattened back to 2D:
             x = x.flatten(start_dim=1)

        A clean way to handle this: check isinstance(layer, ConvPoolLayer)
        and isinstance(layer, FullyConnectedLayer) to decide when to reshape."""
        if isinstance(self.layers[0], ConvPoolLayer):
            x = x.view(x.size(0), 1, 28, 28)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, FullyConnectedLayer) and isinstance(self.layers[i-1], ConvPoolLayer):
                x = x.flatten(start_dim=1)
            x = layer(x)
        return x


    def train_network(self, training_data, epochs, mini_batch_size, eta,
                      lmbda=0.0, test_data=None):
        """Train using mini-batch SGD with cross-entropy loss.

        Key differences from network 1 & 2:

        Loss function:
          Use nn.CrossEntropyLoss (not MSELoss). CrossEntropyLoss expects
          integer class labels (0-9), not one-hot vectors. The training data
          from mnist_loader has one-hot y vectors, so extract the integer
          label with: torch.argmax(y).

        L2 regularisation:
          Pass weight_decay=lmbda/len(training_data) to optim.SGD instead
          of computing the penalty manually as in network2.

        Dropout:
          Call self.train() before training and self.eval() before evaluate()
          so that dropout is active during training and disabled during eval.

        ``lmbda`` — L2 regularisation strength (0 = no regularisation)"""
        optimizer = optim.SGD(self.parameters(), lr=eta,
                              weight_decay=lmbda / len(training_data) if lmbda else 0)
        loss_fn = nn.CrossEntropyLoss()
        n = len(training_data)

        for j in range(epochs):
            self.train()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                xs = torch.stack([torch.tensor(x, dtype=torch.float32).squeeze() for x, _ in mini_batch])
                ys = torch.stack([torch.argmax(torch.tensor(y, dtype=torch.float32)) for _, y in mini_batch])
                optimizer.zero_grad()
                output = self.forward(xs)
                loss = loss_fn(output, ys)
                loss.backward()
                optimizer.step()

            self.eval()
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        """Return the number of test inputs correctly classified."""
        with torch.no_grad():
            test_results = [(torch.argmax(self.forward(torch.tensor(x, dtype=torch.float32).unsqueeze(0))), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) # Return the number of correct results
