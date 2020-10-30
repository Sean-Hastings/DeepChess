import numpy as np
import torch
from torch.utils.data import Dataset


class AESet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).type(torch.FloatTensor)
        return (data, data)

    def __len__(self):
        return self.data.shape[0]


class SiameseSet(Dataset):
    '''
    The labels will indicate which sample is from set B
    '''
    def __init__(self, set_a, set_b, epoch_length):
        self.set_a  = set_a
        self.set_b  = set_b
        self.length = epoch_length

    def __getitem__(self, index):
        #i_a = index % self.set_a.shape[0]
        #i_b = index // self.set_a.shape[0]
        i_a = np.random.randint(0, len(self.set_a))
        i_b = np.random.randint(0, len(self.set_b))

        sample_a = self.set_a[i_a]
        sample_b = self.set_b[i_b]
        samples  = (sample_a, sample_b)

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
