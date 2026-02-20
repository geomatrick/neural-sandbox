# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python sandbox for learning neural networks by implementing MNIST digit classification two ways:
1. **NumPy** — manual backpropagation following Michael Nielsen's "Neural Networks and Deep Learning"
2. **PyTorch** — the same network reimplemented using PyTorch idioms (`nn.Module`, autograd, optimiser)

## Structure

All code lives in the `mnist/` directory:

- [mnist/mnist_network_numpy.py](mnist/mnist_network_numpy.py) — NumPy implementation with manual backprop
- [mnist/mnist_network_pytorch.py](mnist/mnist_network_pytorch.py) — PyTorch reimplementation
- [mnist/mnist_loader.py](mnist/mnist_loader.py) — loads `data/mnist.pkl.gz` via `load_data_wrapper()`
- [mnist/expand_mnist.py](mnist/expand_mnist.py) — augments training set to 250k images by shifting pixels
- [mnist/mnist_network_numpy_train.ipynb](mnist/mnist_network_numpy_train.ipynb) — notebook to train/run NumPy network
- [mnist/mnist_network_pytorch_train.ipynb](mnist/mnist_network_pytorch_train.ipynb) — notebook to train/run PyTorch network

MNIST data is not committed. Download it with:
```bash
mkdir -p mnist/data
curl -o mnist/data/mnist.pkl.gz https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz
```

## Dependencies

```bash
pip install numpy torch
```

## Running

From the `mnist/` directory, launch a notebook:
```bash
jupyter notebook mnist_network_numpy_train.ipynb
```

Or from the repo root via Python:
```python
import sys; sys.path.insert(0, 'mnist')
import mnist_loader, mnist_network_numpy
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = mnist_network_numpy.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

## Architecture

### NumPy network (`Network` class)
- `SGD(training_data, epochs, mini_batch_size, eta, test_data)` — outer training loop
- `update_mini_batch(mini_batch, eta)` — accumulates gradients across the batch then applies update
- `backprop(x, y)` — manual forward + backward pass; returns `(nabla_b, nabla_w)`
- `feedforward(a)` — inference only (no gradient tracking)
- `evaluate(test_data)` — argmax of output layer vs label

### PyTorch network (`Network(nn.Module)`)
- `forward(x)` — replaces `feedforward`; called automatically by PyTorch
- `train_network(...)` — replaces `SGD`; uses `optim.SGD`, `nn.MSELoss`, `loss.backward()`
- `evaluate(test_data)` — same logic as NumPy version, wrapped in `torch.no_grad()`
- Weights/biases managed automatically by `nn.Linear` (Kaiming uniform init by default)

### Key difference
The NumPy version implements backpropagation by hand (BP1–BP4 from Nielsen). The PyTorch version replaces the entire `backprop` method with a single `loss.backward()` call.
