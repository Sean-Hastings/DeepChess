from models.autoencoder import AE
from utils import bitboard_from_byteboard, AESet
import numpy as np
import torch
import argparse
import h5py
import torch.utils.data

def featurize(game, model, device):
    with torch.no_grad():
        _, enc = model(game.to(device))
        return enc.detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--model_id', type=str, default='', metavar='N',
                        help='name of the saved model to pull weights from (default: )')
    parser.add_argument('--batch_size', type=int, default=4096, metavar='N',
                        help='size of batches (default: 4096)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N',
                        help='size of batches (default: 16)')
    args = parser.parse_args()

    cuda   = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else "cpu")

    model = AE().to(device)
    state = torch.load('checkpoints/autoencoder/{}/best.pth.tar'.format(args.model_id), map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    model.eval()

    with h5py.File('data/{}/features.hdf5'.format(args.dataset), 'w') as f_out:
        for group in ['train', 'test']:
            out_group = f_out.create_group(group)
            for dset in ['wins','losses','ties']:
                print('Beginning work on {}/{}'.format(group,dset))
                set_key = '{}/{}'.format(group,dset)
                with h5py.File('data/{}/byteboards.hdf5'.format(args.dataset)) as f_in:
                    games = f_in[set_key]
                    outset = out_group.create_dataset(dset, (len(games), 100))

                dloader = torch.utils.data.DataLoader(AESet(args.dataset, group, dset), batch_size=args.batch_size, pin_memory=cuda, num_workers=args.num_workers)

                for i, (batch, _) in enumerate(dloader):
                    inds = slice(args.batch_size * i, args.batch_size * i + len(batch))
                    outset[inds] = featurize(batch, model, device)
                    print('{} / {}'.format(i+1, len(dloader)), end='\r')
                print('')
