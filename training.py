from nn import *
import torch.nn.functional as F
from parameters import *


# set batchNorm to false if you do not want normalization layers in the network
def initialize_nn(n_layers, batchNorm=True):
    C = torch.rand((vocab_size, n_embd))
    layers = []

    if batchNorm:
        layers.append(Linear(block_size * n_embd, n_hidden))
        layers.append(BatchNorm(n_hidden))
        layers.append(Tanh())

        if n_layers > 2:
            for i in range(n_layers - 2):
                layers.append(Linear(n_hidden, n_hidden))
                layers.append(BatchNorm(n_hidden))
                layers.append(Tanh())

        layers.append(Linear(n_hidden, vocab_size))
        layers.append(BatchNorm(vocab_size))
    else:
        layers.append(Linear(vocab_size * n_embd, n_hidden, bias=True))
        layers.append(Tanh())

        if n_layers > 2:
            for i in range(n_layers - 2):
                layers.append(Linear(n_hidden, n_hidden, bias=True))
                layers.append(Tanh())

        layers.append(Linear(n_hidden, vocab_size, bias=True))

    # make last layer less confident
    with torch.no_grad():
        if batchNorm:
            layers[-1].gamma *= 0.1
        else:
            layers[-1].weights *= 0.1
            layers[-1].bias = 0

    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weights *= 5/3  # multiply by 'gain' for tanh()

    parameters = [C] + [p for layer in layers for p in layer.params()]
    for p in parameters:
        p.requires_grad = True
    print("Number of parameters of neural network: ", sum(p.nelement() for p in parameters))

    return C, layers


# C is character-embedding matrix, 'layers' is list of layers
def train(C, layers, steps, Xtr, Ytr):
    parameters = [C] + [p for layer in layers for p in layer.params()]
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
        Xbatch = Xtr[ix]
        Ybatch = Ytr[ix]

        # forward pass
        embedded = C[Xbatch]
        x = embedded.view(embedded.shape[0], -1)
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Ybatch)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        if i % 10000 == 0:  # print every once in a while
            print(f'{i:7d} /{steps:7d}: {loss.item():.4f}')

        if i > 1000:
            break




