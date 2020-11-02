import numpy as np
import torch
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
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data, _ = bitboard_from_byteboard(self.data[index].reshape([-1, self.data.shape[1]]))
        data = torch.from_numpy(np.squeeze(data)).float()
        return (data, data)

    def __len__(self):
        return self.data.shape[0]


class SiameseSet(Dataset):
    '''
    The labels will indicate which sample is from set B
    '''
    def __init__(self, set_a, set_b, epoch_length, byteboards=True):
        self.set_a  = set_a
        self.set_b  = set_b
        self.length = epoch_length
        self.byteboards = byteboards

    def __getitem__(self, index):
        i_a = np.random.randint(0, self.set_a.shape[0])
        i_b = np.random.randint(0, self.set_b.shape[0])

        sample_a   = self.set_a[i_a]
        sample_b   = self.set_b[i_b]
        samples    = (sample_a, sample_b)
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
        #return self.set_a.shape[0] * self.set_b.shape[0]
        return self.length
