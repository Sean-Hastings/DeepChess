from models.autoencoder import AE
from utils import bitboard_from_byteboard
import numpy as np
import torch
import argparse
import h5py


def featurize(game, index, max_indices, model):
    game, _ = bitboard_from_byteboard(game)
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor).to(device))
    print('{} / {}'.format(index, max_indices), end='\r')
    return enc.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--model_id', type=str, default='', metavar='N',
                        help='name of the saved model to pull weights from (default: )')
    parser.add_argument('--batch_size', type=int, default=4096, metavar='N',
                        help='size of batches (default: 4096)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE().to(device)
    state = torch.load('checkpoints/autoencoder/{}/best.pth.tar'.format(args.model_id), map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    model.eval()

    with h5py.File('data/{}/byteboards.hdf5'.format(args.dataset)) as f_in:
        with h5py.File('data/{}/features.hdf5'.format(args.dataset), 'w') as f_out:
            for group in ['train', 'test']:
                out_group = f_out.create_group(group)
                for dset in ['wins','losses','ties']:
                    games = f_in['{}/{}'.format(group,dset)]
                    outset = out_group.create_dataset(dset, (len(games), 100))

                    num_batches = games.shape[0] // args.batch_size
                    inds        = [slice(args.batch_size*i, args.batch_size*(i+1)) for i in range(num_batches)] + [slice(args.batch_size*num_batches, len(games))]

                    for i, batch in enumerate(inds):
                        outset[batch] = featurize(games[batch], i+1, len(inds), model)
