import torch
from torch import nn
from torch.nn import functional as F


def redrop(x, dropout):
    return dropout(x+1)-1


class ReDropout(nn.Module):
    def __init__(self, rate):
        super(ReDropout, self).__init__()
        self.rate = rate
        self.dropout = nn.Dropout(rate)

    def forward(self, x):
        return redrop(x, self.dropout)


class NetworkBlock(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ELU):
        super(NetworkBlock, self).__init__()
        self.linear     = nn.Linear(in_size, out_size)
        self.bn         = nn.BatchNorm1d(out_size)
        self.activation = activation() if activation is not None else None

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Network(nn.Module):
    def __init__(self, shapes, activation=nn.ELU, dropout=0.0):
        super(Network, self).__init__()
        self.shapes     = shapes
        self.activation = activation
        self.dropout    = ReDropout(dropout) if activation is nn.ELU else nn.Dropout(dropout)

        self.modules = []
        activations  = [activation] * (len(shapes) - 2) + [None]
        for from_size, to_size, act in zip(list(shapes)[:-1], list(shapes)[1:], activations):
            self.modules += [NetworkBlock(from_size, to_size, act)]

        self.modules = self.modules[:-1] + [self.dropout] + self.modules[-1:]
        self.net     = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.net(x)
