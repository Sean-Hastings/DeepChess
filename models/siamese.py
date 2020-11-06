import torch
from torch import nn
from torch.nn import functional as F

from .utils import Network
from .autoencoder import AE


class Siamese(nn.Module):
    def __init__(self, hidden_size=(400, 200, 100), dropout=0.0, AE_init=None):
        super(Siamese, self).__init__()
        self.dropout     = dropout
        self.hidden_size = list(hidden_size)
        self.shapes      = self.hidden_size + [2]

        if AE_init is None:
            self.encoder = AE().encoder
        else:
            ae_model  = AE()
            load_dir  = 'checkpoints/autoencoder/{}'.format(AE_init)
            load_path = load_dir + '/best.pth.tar'
            state     = torch.load(load_path, map_location=lambda storage, loc: storage)

            ae_model.load_state_dict(state['state_dict'])
            self.encoder = ae_model.encoder

        self.shapes = [self.encoder.shapes[-1]*2] + self.shapes

        self.net = Network(self.shapes, dropout=self.dropout)

    def forward(self, x):
        state_size = x.shape[1] // 2
        x = torch.cat([self.encoder(x[:, :state_size]),
                       self.encoder(x[:, state_size:])], dim=-1)
        x = self.net(x)
        return torch.softmax(x, dim=-1)

    def name(self):
        return 'siamese'


class SplitSiamese(nn.Module):
    def __init__(self, hidden_size=(200, 100, 50), dropout=0.0):
        super(SplitSiamese, self).__init__()
        self.dropout     = dropout
        self.hidden_size = list(hidden_size)
        self.shapes      = [773] + self.hidden_size + [1]

        self.net = Network(self.shapes, dropout=self.dropout)

    def evaluate_state(self, x):
        return self.net(x)

    def forward(self, x):
        state_size = x.shape[1] // 2
        x = torch.cat([self.net(x[:, :state_size]),
                       self.net(x[:, state_size:])], dim=-1)
        return torch.softmax(x, dim=-1)

    def name(self):
        return 'split_siamese'
