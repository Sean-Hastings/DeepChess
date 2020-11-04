from models.autoencoder import AE
from utils import bitboard_from_byteboard
import numpy as np
import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--model_id', type=str, default='', metavar='N',
                        help='name of the saved model to pull weights from (default: )')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE().to(device)
    model.eval()
    state = torch.load('checkpoints/autoencoder/{}/best.pth.tar'.format(args.model_id), map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    games = np.load('data/{}_byteboards.npy'.format(args.dataset))

    batch_size  = 4096
    num_batches = games.shape[0] // batch_size
    extras      = games[batch_size*num_batches:]

    games = np.split(games[:batch_size*num_batches], num_batches) + [extras]

    def featurize(game, index, max_indices):
        with torch.no_grad():
            game, _ = bitboard_from_byteboard(game)
            _, enc = model(torch.from_numpy(game).type(torch.FloatTensor).to(device))
            print('{} / {}'.format(index, max_indices), end='\r')
            return enc.cpu().detach().numpy()

    for i in range(len(games)):
        games[i] = featurize(games[i], i+1, num_batches)

    #games = [featurize(batch, i+1, len(games)) for i, batch in enumerate(games)]
    games = np.concatenate(games, axis=0)

    np.save('./data/{}_features.npy'.format(args.dataset), games)
