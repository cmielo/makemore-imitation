import torch


class Linear:
    def __init__(self, n_in, n_out, bias=False):
        # divide by sqrt(n_in) so that values in 'weights' are more squashed
        self.weights = torch.randn((n_in, n_out)) / n_in ** 0.5
        self.bias = torch.randn(n_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def params(self):
        return [self.weights] + (self.bias if self.bias is not None else [])


class BatchNorm:
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
        self.std_running = torch.ones(dimension)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xstd = x.std(0, keepdim=True)
        else:
            xmean = self.mean_running
            xstd = self.std_running

        # normalization of a batch,
        # -> norm := (X - μ)/σ so that it follows ~ N(0, 1)
        # -> result := gamma * norm + bias, so that neural network can make results more defuse/sharp/biased
        normalized = (x - xmean) / (xstd + self.e)  # add 'e' so that you do not divide by 0 accidentally

        self.out = self.gamma * normalized + self.beta

        if self.training:
            with torch.no_grad():
                # dynamically update mean and std if during training
                self.mean_running = (1 - self.momentum) * self.mean_running + self.momentum * xmean
                self.std_running = (1 - self.momentum) * self.std_running + self.momentum * xstd
        return self.out

    def params(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def params(self):
        return []
