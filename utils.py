import numpy as np
import torch
import h5py
from itertools import accumulate
from time import time
from torch.utils.data import Dataset


def bitboard_from_byteboard(byteboard):
    '''
    expects batched inputs (and therefore outputs)
    '''
    bitboard = np.zeros((byteboard.shape[0], 64*6*2+5))
    extras   = byteboard[:, -1]

    byte_b = byteboard[:, :-1] >> 4
    byte_a = byteboard[:, :-1] - (byte_b << 4)
    byteboard = np.stack([byte_a, byte_b], axis=-1).reshape(-1, 64) - 1

    mask    = byteboard < 255
    color   = byteboard // 6 + 1
    i_piece = byteboard % 6
    indices = np.arange(byteboard.shape[1]).reshape([1, -1])
    pieces  = (i_piece + indices * 6) * color
    indices = np.stack([np.arange(byteboard.shape[0])] * byteboard.shape[1], axis=-1)

    bitboard[indices[mask], pieces[mask]] = 1
    bitboard[:, -5] = ((extras << 1) >> 7)
    bitboard[:, -4] = ((extras << 2) >> 7)
    bitboard[:, -3] = ((extras << 3) >> 7)
    bitboard[:, -2] = ((extras << 4) >> 7)
    bitboard[:, -1] = ((extras << 5) >> 7)

    labels = ((extras << 6) >> 6)
    labels[labels == 1] = -1
    labels[labels == 2] = 1

    return bitboard, labels


class AESet(Dataset):
    def __init__(self, path, modes, dsets):
        self.dataset_path = path
        self.modes = modes if isinstance(modes, (list, tuple)) else [modes]
        self.dsets = dsets if isinstance(dsets, (list, tuple)) else [dsets]
        self.length = dict()
        with h5py.File('data/{}/byteboards.hdf5'.format(self.dataset_path)) as f:
            for mode in self.modes:
                for dset in self.dsets:
                    address = '{}/{}'.format(mode, dset)
                    self.length[address] = len(f[address])


        self.length['all'] = sum([self.length[key] for key in self.length.keys()])

    def __getitem__(self, index):
        with h5py.File('data/{}/byteboards.hdf5'.format(self.dataset_path)) as f:
            keys = list(self.length.keys())
            ranges = list(accumulate([0] + [self.length[key] for key in keys]))
            for i in range(len(keys)):
                if index < ranges[i+1]:
                    key = keys[i]
                    k_index = index - ranges[i]
                    break

            data = f[key][k_index]
            data, _ = bitboard_from_byteboard(data.reshape([-1, data.shape[-1]]))
            data = torch.from_numpy(np.squeeze(data)).float()
            return (data, data)

    def __len__(self):
        return self.length['all']


class SiameseSet(Dataset):
    '''
    The labels will indicate which sample is from set B
    '''
    def __init__(self, dataset, mode, epoch_length):
        self.dataset_path = dataset
        self.mode = mode
        with h5py.File('data/{}/byteboards.hdf5'.format(self.dataset_path)) as f:
            f = f[self.mode]
            self.length = {'wins':f['wins'].len(), 'losses':f['losses'].len(), 'ties':f['ties'].len(), 'all':0}
            self.length['all'] = epoch_length

    def __getitem__(self, index):
        i_win  = np.random.randint(0, self.length['wins'])
        i_loss = np.random.randint(0, self.length['losses'])

        with h5py.File('data/{}/byteboards.hdf5'.format(self.dataset_path)) as f:
            f = f[self.mode]
            sample_win  = f['wins'][i_win]
            sample_loss = f['losses'][i_loss]
            samples     = (sample_loss, sample_win)

        if self.byteboards:
            samples, _ = bitboard_from_byteboard(np.stack(samples, axis=0))

        order   = np.random.randint(0,2)
        i_o     = (order, 1-order)
        samples = (samples[i_o[0]], samples[i_o[1]])
        stacked = np.hstack(samples)
        stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
        label   = torch.from_numpy(np.array(i_o)).type(torch.FloatTensor)

        return (stacked, label)

    def __len__(self):
        return self.length['all']
