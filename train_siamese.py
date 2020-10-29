import numpy as np
from torch.nn import functional as F

from utils import SiameseSet
from models.siamese import Siamese
from train import train


def loss_function(predictions, labels):
    BCE = F.binary_cross_entropy(predictions, labels, reduction='sum')
    return BCE


def get_acc(predictions, labels):
    correct = ((predictions > .5) * labels).sum()
    return correct


if __name__ == '__main__':
    print('Loading data...')
    games = np.load('./data/features.npy')
    wins  = np.load('./data/labels.npy')
    games = games[wins != 0][:20000]
    wins  = wins[wins != 0][:20000]
    print('processing data...')

    # TODO: explain this mess
    test_percent = 0.1
    num_test     = int(len(games)*test_percent)
    test_games   = games[:num_test//2]
    test_wins    = wins[:num_test//2]
    games        = games[num_test//2:]
    wins         = wins[num_test//2:]
    p     = np.random.permutation(len(wins))
    games = games[p]
    wins  = wins[p]
    train_games = games[num_test//2:]
    train_wins  = wins[num_test//2:]
    test_games  = np.concatenate([test_games, games[:num_test//2]])
    test_wins   = np.concatenate([test_wins, wins[:num_test//2]])

    train_games_wins = train_games[train_wins == 1]
    train_games_losses = train_games[train_wins == -1]

    test_games_wins = test_games[test_wins == 1]
    test_games_losses = test_games[test_wins == -1]
    train_set = SiameseSet(train_games_losses, train_games_wins)
    test_set  = SiameseSet(test_games_losses, test_games_wins)

    model = Siamese

    test_functions = {'accuracy': get_acc}

    train(train_set, test_set, model, loss_function, test_functions)
