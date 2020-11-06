import argparse
from time import time
from collections import defaultdict
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
from tensorboardX import SummaryWriter


def train_epoch(epoch, model, optimizer, loss_function, data_loader, writer, device, args):
    model.train()
    train_loss = 0
    logtime = time() - args.log_interval
    for batch_idx, (data, labels) in enumerate(data_loader):
        data   = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (time() - logtime) > args.log_interval:
            logtime = time()
            ib = batch_idx + 1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, ib * len(data), len(data_loader.dataset),
                100. * ib / len(data_loader),
                loss.item() / len(data)))
            writer.add_scalar('data/train_loss', loss.item() / len(data), (epoch-1)*len(data_loader.dataset) + ib * args.batch_size)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))
    return train_loss / len(data_loader.dataset)


def test(epoch, model, loss_function, data_loader, writer, device, args, test_functions={}):
    model.eval()
    test_loss = 0
    test_metrics = defaultdict(lambda: 0)
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            predictions = model(data)
            loss = loss_function(predictions, labels)
            test_loss += loss.item()

            for key in list(test_functions.keys()):
                test_metrics[key] += test_functions[key](predictions, labels)

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('data/test_loss', test_loss, epoch)

    for key in list(test_metrics.keys()):
        metric = test_metrics[key] / len(data_loader.dataset)
        print('====> Test set {}: {:.4f}'.format(key, metric))
        writer.add_scalar('data/test_{}'.format(key), metric, epoch)

    return test_loss


def save(epoch, model, optimizer, loss, args, best=False):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1,
             'test_loss': loss}
    save_dir = 'checkpoints/{}/{}'.format(model.name(), args.id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, '{}.pth.tar'.format(epoch)))
    if best:
        torch.save(state, os.path.join(save_dir, 'best.pth.tar'))


def train(train_set, test_set, model, loss_function, test_functions={}, model_kwargs={}):
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='dropout for each layer during training (default: 0.0)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='N',
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--decay', type=float, default=.99, metavar='N',
                        help='decay rate of learning rate (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--num_workers', type=int, default=8, metavar='S',
                        help='number of dataloader workers (default: 8)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--id', type=str, default='', metavar='N',
                        help='unique identifier for saving weights')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many seconds to wait before logging training status (default: 20)')
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    if len(args.id) == 0:
        args.id = 'lr_{}-decay_{}-batchsize_{}'.format(args.lr, args.decay, args.batch_size)

    writer = SummaryWriter(comment='_' + args.id)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    model_kwargs['dropout'] = args.dropout
    model = model(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    start_epoch = 1
    best_loss = float('inf')
    try:
        load_dir = 'checkpoints/{}/{}'.format(model.name(), args.id)
        load_path = load_dir + '/best.pth.tar'
        state = torch.load(load_path,
                            map_location=lambda storage, loc: storage)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        best_loss = state['test_loss']
    except Exception:
        pass

    print('Beginning training:')

    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(epoch, model, optimizer, loss_function, train_loader, writer, device, args)
        tl = test(epoch, model, loss_function, test_loader, writer, device, args, test_functions)
        save(epoch, model, optimizer, tl, args, tl < best_loss)
        best_loss = min(best_loss, tl)

        # Adjust learning rate
        for params in optimizer.param_groups:
            params['lr'] *= args.decay


if __name__ == '__main__':
    print('Try train_autoencoder.py or train_siamese.py to start training!')
