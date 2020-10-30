import numpy as np
from torch.nn import functional as F

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
    print('Loading data...')
    games  = np.load('data/bitboards.npy')
    labels = np.load('data/labels.npy')
    print('Pruning data...')
    games = games[labels != 0]
    del labels
    print('Processing data...')


    # TODO: explain this mess
    test_percent = 0.1
    num_test = int(len(games)*test_percent)
    test_games = games[:num_test]
    games = games[num_test:]
    train_set = AESet(train_games)
    test_set  = AESet(test_games)

    model = AE

    test_functions = {'diff': diff,
                      'mse': mse_loss_function}

    train(train_set, test_set, model, loss_function, test_functions)
