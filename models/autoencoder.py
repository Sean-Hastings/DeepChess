import torch
from torch import nn
from torch.nn import functional as F

from .utils import Network

class AE(nn.Module):
    def __init__(self, hidden_size=(600, 400, 200, 100), dropout=0.0):
        super(AE, self).__init__()
        self.dropout     = nn.Dropout(dropout)
        self.hidden_size = list(hidden_size)

        self.encoder = Network([773] + self.hidden_size, dropout=self.dropout)
        self.decoder = Network(self.hidden_size[::-1] + [773])

    def encode(self, x):
        x = torch.tanh(self.encoder(x))
        return x

    def decode(self, z):
        z = torch.sigmoid(self.decoder(z))
        return z

    def forward(self, x):
        enc = self.encode(x)
        return self.decode(enc), enc

    def name(self):
        return 'autoencoder'
