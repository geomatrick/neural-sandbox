#### Libraries
import random
import torch
import torch.nn as nn
import torch.optim as optim

"""Here, I am taking the Nielsen Mnist network that was written with Numpy and re-writing it in Pytorch to learn how Pytorch works"""


class Network(nn.Module):

    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])]) # this includes the weight and biases e.g. net.layers[0].weight.shape

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x

    def train_network(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        optimizer = optim.SGD(self.parameters(), lr=eta)
        loss_fn = nn.MSELoss()
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]


            for mini_batch in mini_batches:
                xs = torch.stack([torch.tensor(x, dtype=torch.float32).squeeze() for x, _ in mini_batch])
                ys = torch.stack([torch.tensor(y, dtype=torch.float32).squeeze() for _, y in mini_batch])
                optimizer.zero_grad()
                output = self.forward(xs)
                loss = loss_fn(output, ys)
                loss.backward()
                optimizer.step()
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        with torch.no_grad():
            test_results = [(torch.argmax(self.forward(torch.tensor(x, dtype=torch.float32).squeeze())), y)
                for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) # Return the number of correct results

