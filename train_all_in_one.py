import numpy as np
from torch.nn import functional as F

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

    print('Loading data...')
    games = np.load('data/{}_byteboards.npy'.format(args.dataset))
    wins  = (games[:, -1] << 6) >> 6
    games = games[wins != 0]
    wins  = wins[wins != 0]
    print('processing data...')

    # TODO: explain this mess
    test_percent = 0.01
    num_test     = int(len(games)*test_percent)
    test_games   = games[:num_test]
    test_wins    = wins[:num_test]
    train_games  = games[num_test:]
    train_wins   = wins[num_test:]

    train_games_wins   = train_games[train_wins == 2]
    train_games_losses = train_games[train_wins == 1]
    test_games_wins    = test_games[test_wins == 2]
    test_games_losses  = test_games[test_wins == 1]

    print('Training set consists of {} win-states and {} lose-states'.format(train_games_wins.shape[0], train_games_losses.shape[0]))
    print('Testing set consists of {} win-states and {} lose-states'.format(test_games_wins.shape[0], test_games_losses.shape[0]))

    train_set = SiameseSet(train_games_losses, train_games_wins, 1000000)
    test_set  = SiameseSet(test_games_losses, test_games_wins, 50000)

    model = SplitSiamese

    test_functions = {'accuracy': get_acc}

    train(train_set, test_set, model, loss_function, test_functions)
