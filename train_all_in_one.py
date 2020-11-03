import numpy as np
from torch.nn import functional as F
import argparse

from utils import SiameseSet
from models.siamese import SplitSiamese
from train import train


def loss_function(predictions, labels):
    BCE = F.binary_cross_entropy(predictions, labels, reduction='sum')
    return BCE


def get_acc(predictions, labels):
    correct = ((predictions > .5) * labels).sum()
    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a split-siamese model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    args, _ = parser.parse_known_args()

    train_set = SiameseSet(args.dataset, 'train', 1000000)
    test_set  = SiameseSet(args.dataset, 'test', 50000)

    model = SplitSiamese

    test_functions = {'accuracy': get_acc}

    try:
        train(train_set, test_set, model, loss_function, test_functions)
    finally:
        train_set.close()
        test_set.close()
