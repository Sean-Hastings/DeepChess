import torch
from torch import nn
from torch.nn import functional as F

from autoencoder import redrop

class Siamese(nn.Module):
    def __init__(self, dropout=0.0):
        super(Siamese, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(200, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = redrop(F.elu(self.bn1(self.fc1(x))), self.dropout)
        x = redrop(F.elu(self.bn2(self.fc2(x))), self.dropout)
        x = redrop(F.elu(self.bn3(self.fc3(x))), self.dropout)
        x = torch.sigmoid(self.fc4(x))
        return x

    def name(self):
        return 'siamese'
