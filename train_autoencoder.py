import numpy as np
import torch
from torch.nn import functional as F
import argparse

from utils import AESet
from models.autoencoder import AE
from train import train


def loss_function(predictions, labels):
    BCE = F.binary_cross_entropy(predictions[0], labels, reduction='sum')
    return BCE


def mse_loss_function(predictions, labels):
    MSE = F.mse_loss(predictions[0], labels, reduction='sum')
    return MSE

def diff(predictions, labels):
    diff = (predictions[0] - labels).abs().sum()
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an autoencoder model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    args, _ = parser.parse_known_args()

    print('Loading data...')
    games = np.load('data/{}_byteboards.npy'.format(args.dataset))
    wins  = (games[:, -1] << 6) >> 6
    games = games[wins != 0]
    del wins
    print('Processing data...')


    # TODO: explain this mess
    test_percent = 0.01
    num_test = int(len(games)*test_percent)
    test_games = games[:num_test]
    train_games = games[num_test:]
    p = torch.randperm(train_games.shape[0])
    train_set = AESet(train_games[p])
    test_set  = AESet(test_games)

    model = AE

    test_functions = {'diff': diff,
                      'mse': mse_loss_function}

    train(train_set, test_set, model, loss_function, test_functions)
