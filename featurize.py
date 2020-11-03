from models.autoencoder import AE
from utils import bitboard_from_byteboard
import numpy as np
import torch
import argparse


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

    with h5py.File('data/{}/byteboards.hdf5'.format(self.dataset)) as f_in:
        with h5py.File('data/{}/features.hdf5'.format(self.dataset), 'w') as f_out:
            for group in ['train', 'test']:
                out_group = f_out.create_group(group)
                for dset in ['wins','losses','ties']for dset in ['{}/{}'.format(a,b) for a in ['train','test'] for b in ['wins','losses','ties']]:
                    games = f_in['{}/{}'.format(group,dset)]

                    num_batches = games.shape[0] // args.batch_size
                    extras      = games[args.batch_size*num_batches:]

                    batched_games = np.split(games[:args.batch_size*num_batches], num_batches) + [extras]

                    feat_games = [featurize(batch, i+1, len(batched_games), model) for i, batch in enumerate(batched_games)]
                    featurized = np.vstack(feat_games)

                    out_group.create_dataset(dset, data=featurized)
