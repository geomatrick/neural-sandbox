# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python sandbox for implementing a feedforward neural network trained with stochastic gradient descent (SGD) and backpropagation, targeting MNIST digit classification. Based on Michael Nielsen's "Neural Networks and Deep Learning" book.

## Running the Code

```bash
python mnist-network.py
```

Dependencies: `numpy` (install via `pip install numpy`).

## Architecture

Single module: [mnist-network.py](mnist-network.py)

- `Network(sizes)` — core class; `sizes` is a list of layer widths (e.g. `[784, 30, 10]` for MNIST)
- `SGD(training_data, epochs, mini_batch_size, eta, test_data)` — outer training loop
- `update_mini_batch(mini_batch, eta)` — applies one gradient descent step via backprop
- `backprop(x, y)` — returns `(nabla_b, nabla_w)` gradients using forward + backward pass
- `feedforward(a)` — inference pass; applies sigmoid at each layer
- `evaluate(test_data)` — counts correct classifications (argmax of output layer)
- `sigmoid(z)` / `sigmoid_prime(z)` — activation function and its derivative

## Current Status

All methods are stubs. The docstrings describe the expected behavior; implementations need to be filled in.
