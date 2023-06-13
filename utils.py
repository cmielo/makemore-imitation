import torch.nn.functional as F
from parameters import *
from nn import *


def initialize():
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])

    with torch.no_grad():
        # last layer: make less confident
        model.layers[-1].weights *= 0.1
        # all other linear layers: apply gain (https://pytorch.org/docs/stable/nn.init.html)
        for layer in model.layers[:-1]:
            if isinstance(layer, Linear):
                layer.weights *= 5 / 3

    params = model.parameters()
    print(sum(p.nelement() for p in params))  # number of parameters in total
    for p in params:
        p.requires_grad = True
    return model


def train(model, steps, Xtr, Ytr):
    parameters = model.parameters()
    for i in range(steps):
        # sample the batch
        indexes = torch.randint(0, Xtr.shape[0], (batch_size, ))
        Xbatch = Xtr[indexes]
        Ybatch = Ytr[indexes]

        # forward pass
        logits = model(Xbatch)
        loss = F.cross_entropy(logits, Ybatch)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        if i % 10000 == 0:  # print loss value every 10000 iterations while
            print(f'{i:7d} /{steps:7d}: {loss.item():.4f}')

        # if i > 3000: # remove if full run is desired
        #     break


class Utils:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.model = None

    def build_dataset(self, words):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def set_model(self, model):
        self.model = model

    @torch.no_grad()  # this decorator disables gradient tracking
    def split_loss(self, dictionary, split):
        x, y = dictionary[split]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        print(split, loss.item())

    def sample(self):
        for _ in range(20):

            out = []
            context = [0] * block_size  # initialize with all ...
            while True:
                # forward pass the neural net
                logits = self.model(torch.tensor([context]))
                probs = F.softmax(logits, dim=1)
                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1).item()
                # shift the context window and track the samples
                context = context[1:] + [ix]
                out.append(ix)
                # if we sample the special '.' token, break
                if ix == 0:
                    break

            print(''.join(self.itos[i] for i in out))  # decode and print the generated word




