import random
from training import *

def build_dictionary(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


if __name__ == '__main__':
    words = open("names.txt", "r").read().splitlines()
    stoi, itos = build_dictionary(words)
    random.shuffle(words)
    l1 = int(0.8*len(words))
    l2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:l1])  # training set
    Xdev, Ydev = build_dataset(words[l1:l2])  # dev set
    Xte, Yte = build_dataset(words[l2:])  # test set

    C, layers = initialize_nn(3)
    train(C, layers, 10000, Xtr, Ytr)

