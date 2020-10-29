import torch
from torch import nn
from torch.nn import functional as F

#from autoencoder import redrop
def redrop(x, dropout):
    return dropout(x)#+1)-1

class Siamese(nn.Module):
    def __init__(self, dropout=0.0):
        super(Siamese, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(200, 2)

    def forward(self, x):
        x = torch.softmax(self.fc1(x), dim=-1)
        return x

    def name(self):
        return 'siamese'
