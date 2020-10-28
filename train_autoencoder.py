from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from tensorboardX import SummaryWriter

from models.autoencoder import AE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                    help='dropout for each AE layer during training (default: 0.0)')
parser.add_argument('--lr', type=float, default=5e-3, metavar='N',
                    help='learning rate (default: 5e-3)')
parser.add_argument('--decay', type=float, default=.95, metavar='N',
                    help='decay rate of learning rate (default: 0.95)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--id', type=str, default='', metavar='N',
                    help='unique identifier for saving weights')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if len(args.id) > 0:
    args.id = '_' + args.id

lr = args.lr
decay = args.decay
batch_size = args.batch_size

writer = SummaryWriter()#comment='lr: {} | decay: {} | batch size: {}'.format(lr, decay, batch_size))
print('Loading and processing data...')

games = np.load('data/bitboards.npy')

# TODO: explain this mess
test_percent = 0.1
num_test = int(len(games)*test_percent)
test_games = games[:num_test//2]
games = games[num_test//2:]
np.random.shuffle(games)
train_games = games[num_test//2:]
test_games  = np.concatenate([test_games, games[:num_test//2]])

class TrainSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return (torch.from_numpy(train_games[index]).type(torch.FloatTensor), 1)

    def __len__(self):
        return train_games.shape[0]

class TestSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return (torch.from_numpy(test_games[index]).type(torch.FloatTensor), 1)

    def __len__(self):
        return test_games.shape[0]

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 773), reduction='sum')
    return BCE

def mse_loss_function(recon_x, x):
    MSE = F.mse_loss(recon_x, x.view(-1, 773), reduction='sum')
    return MSE

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, enc = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        ib = batch_idx + 1
        if ib % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, ib * len(data), len(train_loader.dataset),
                100. * ib / len(train_loader),
                loss.item() / len(data)))
            writer.add_scalar('data/train_loss', loss.item() / len(data), epoch*len(train_loader) + batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_mse = 0
    total_diff = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, enc = model(data)
            pred = (recon_batch.cpu().detach().numpy() > .5).astype(int)
            total_diff += float(np.sum(data.cpu().detach().numpy() != pred))
            test_loss += loss_function(recon_batch, data).item()
            test_loss_mse += mse_loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    test_loss_mse /= len(test_loader.dataset)
    total_diff /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set loss (mse): {:.4f}'.format(test_loss_mse))
    print('====> Test set diff: {:.4f}'.format(total_diff))
    writer.add_scalar('data/test_loss', test_loss, epoch)
    writer.add_scalar('data/test_loss_mse', test_loss_mse, epoch)
    writer.add_scalar('data/test_diff', total_diff, epoch)

    return test_loss

def save(epoch, best=False):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = 'checkpoints/autoencoder/lr_{}_decay_{}{}'.format(int(lr*1000), int(decay*100), args.id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, 'ae_{}.pth.tar'.format(epoch)))
    if best:
        torch.save(state, os.path.join(save_dir, 'ae_best.pth.tar'))

def recon(game):
    recon, _ = model(torch.from_numpy(game).type(torch.FloatTensor))
    recon = (recon.cpu().detach().numpy() > .5).astype(int)
    return recon


train_loader = torch.utils.data.DataLoader(TrainSet(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(), batch_size=batch_size)

model = AE(args.dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

start_epoch = 1
try:
    load_dir = 'checkpoints/autoencoder/lr_{}_decay_{}{}'.format(int(lr*1000), int(decay*100), args.id)
    load_path = load_dir + '/ae_best.pth.tar'
    state = torch.load(load_path,
                        map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
except Exception:
    pass

print('Data ready, beginning training:')

best_loss = float('inf')
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    tl = test(epoch)
    save(epoch, tl < best_loss)
    best_loss = min(best_loss, tl)

    # Adjust learning rate
    for params in optimizer.param_groups:
        params['lr'] *= decay
