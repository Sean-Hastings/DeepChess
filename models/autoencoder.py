import torch
from torch import nn
from torch.nn import functional as F

def redrop(x, dropout):
    return dropout(x)#+1)-1

class AE(nn.Module):
    def __init__(self, dropout=0.0):
        super(AE, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.fce1 = nn.Linear(773, 600)
        self.bne1 = nn.BatchNorm1d(600)
        self.fce2 = nn.Linear(600, 400)
        self.bne2 = nn.BatchNorm1d(400)
        self.fce3 = nn.Linear(400, 200)
        self.bne3 = nn.BatchNorm1d(200)
        self.fce4 = nn.Linear(200, 100)

        self.fcd1 = nn.Linear(100, 200)
        self.bnd1 = nn.BatchNorm1d(200)
        self.fcd2 = nn.Linear(200, 400)
        self.bnd2 = nn.BatchNorm1d(400)
        self.fcd3 = nn.Linear(400, 600)
        self.bnd3 = nn.BatchNorm1d(600)
        self.fcd4 = nn.Linear(600, 773)

    def encode(self, x):
        x = redrop(F.elu(self.bne1(self.fce1(x))), self.dropout)
        x = redrop(F.elu(self.bne2(self.fce2(x))), self.dropout)
        x = redrop(F.elu(self.bne3(self.fce3(x))), self.dropout)
        x = torch.tanh(self.fce4(x))
        return x

    def decode(self, z):
        z = redrop(F.elu(self.bnd1(self.fcd1(z))), self.dropout)
        z = redrop(F.elu(self.bnd2(self.fcd2(z))), self.dropout)
        z = redrop(F.elu(self.bnd3(self.fcd3(z))), self.dropout)
        z = torch.sigmoid(self.fcd4(z))
        return z

    def forward(self, x):
        enc = self.encode(x.view(-1, 773))
        return self.decode(enc), enc

    def name(self):
        return 'autoencoder'
