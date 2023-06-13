import random
from utils import *


def build_dictionary(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


"""
Exemplary approach:
    - Read the data - names in this example
    - Build the dictionary - map chars to ints and the other way around
    - Build the datasets - training set, validation set (dev), test set
    - Define the list of layers in neural network and put it in Sequential()
    - Run the optimization
    - Sample from the model
"""
if __name__ == '__main__':
    words = open("names.txt", "r").read().splitlines()
    stoi, itos = build_dictionary(words)

    util = Utils(stoi, itos)

    random.shuffle(words)
    l1 = int(0.8*len(words))
    l2 = int(0.9*len(words))

    Xtr, Ytr = util.build_dataset(words[:l1])  # training set
    Xdev, Ydev = util.build_dataset(words[l1:l2])  # dev set
    Xte, Yte = util.build_dataset(words[l2:])  # test set

    # initialized neural network
    model = initialize()
    util.set_model(model)

    train(model, number_of_iterations, Xtr, Ytr)

    for layer in model.layers:
        if isinstance(layer, BatchNorm):
            layer.training = False

    dictionary = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }

    option = 0
    while option != 3:
        print("1 - Calculate loss over whole training set and validation set.")
        print("2 - Sample from the model.")
        print("3 - Exit")
        option = int(input())
        match option:
            case 1:
                util.split_loss(dictionary, 'train')
                util.split_loss(dictionary, 'val')
            case 2:
                util.sample()
            case _:
                print("\n")




