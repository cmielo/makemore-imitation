import torch

"""
nn.py contains the definitions of layers that can be passed to a Sequential() class to create a neural network
They are roughly following PyTorch API and some scientific papers
"""


class Linear:
    """
    Linear layer has parameters:
        - weights, which is a tensor in which every column represents the weights of inputs to every neuron
        - bias, which is an optional tensor (when using batch norm it is recommended to be None)
    """
    def __init__(self, n_in, n_out, bias=False):
        # divide by sqrt(n_in) so that values in 'weights' are more squashed
        self.weights = torch.randn((n_in, n_out)) / n_in ** 0.5
        self.bias = torch.randn(n_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weights] + (self.bias if self.bias is not None else [])


class BatchNorm:
    """
    Batch normalization layer, this implementation follows PyTorch API and the paper:
        "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
         which can be found on https://arxiv.org/pdf/1502.03167.pdf
    """
    def __init__(self, dimension, e=1e-5, momentum=0.01):
        # initial gain, all ones, as I initially we do not want it to make any impact on batches
        self.gamma = torch.ones(dimension)
        # initial bias, all zeros, as I initially we do not want it to make any impact on batches
        self.beta = torch.zeros(dimension)
        self.e = e
        self.momentum = momentum
        self.training = True
        # mean of activations that will be calculated alongside the training of nn
        self.mean_running = torch.zeros(dimension)
        # same for standard deviation
        self.var_running = torch.ones(dimension)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0  # dimension for mean for case where x has shape [A, B]
            if x.ndim == 3:
                dim = (0, 1)  # dimensions for case where x has shape [A, B ,C]

            # if dimensionality of x is different from 2 or 3, then we get an error, which is desired
            xmean = x.mean(dim, keepdim=True)  # batch mean for case where x has shape [A, B ,C]
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean = self.mean_running
            xvar = self.var_running

        # normalization of a batch,
        # -> norm := (X - μ)/σ so that it follows ~ N(0, 1)
        # -> result := gamma * norm + bias, so that neural network can make results more defuse/sharp/biased
        normalized = (x - xmean) / torch.sqrt(xvar + self.e)  # add 'e' so that you do not divide by 0 accidentally
        self.out = self.gamma * normalized + self.beta

        if self.training:
            with torch.no_grad():
                # dynamically update mean and std if during training
                self.mean_running = (1 - self.momentum) * self.mean_running + self.momentum * xmean
                self.std_running = (1 - self.momentum) * self.var_running + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    """
    Tanh layer, squashes the outputs of neurons to values in the interval (-1; 1) using the tanh(x) hyperbolic function.
    """
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    """
    Simple embedding layer which embeds the context characters to their vector representation in n_embd dimensions.
    Parameters:
        - weight, which is of size (vocab_size, n_embd)
    """
    def __init__(self, num_embeddings, n_embd):
        self.weight = torch.randn((num_embeddings, n_embd))

    def __call__(self, Xb):
        self.out = self.weight[Xb]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    """
    FlattenConsecutive layer, concatenates the embeddings of context in the way that follows the paper:
        "Wavenet: a generative model for raw audio"
        which can be found on https://arxiv.org/pdf/1609.03499.pdf
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, self.n * C)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    """
    Sequential structure that contains the array of layers in the network.
    It is called while passing a single batch or one context to make a forward pass in the whole network.
    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
