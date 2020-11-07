import numpy as np
import chess
import argparse
import h5py
from time import time
from multiprocessing import Pool
import os


def get_byteboard(board, result):
    byteboard = np.zeros(32 + 1).astype(np.uint8)

    piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

    for i in range(64):
        i_board = i // 2
        i_hbyte = i % 2
        if board.piece_at(i):
            offset = i_hbyte * 4
            color = int(board.piece_at(i).color)
            value = piece_idx[board.piece_at(i).symbol().lower()] + 6 * color + 1
            byteboard[i_board] += value << offset

    extras = [board.turn,
              board.has_kingside_castling_rights(True),
              board.has_kingside_castling_rights(False),
              board.has_queenside_castling_rights(True),
              board.has_queenside_castling_rights(False)]
    byteboard[-1] = np.sum([int(x) << (i+2) for i, x in enumerate(extras)]+[result])

    return byteboard


def get_result(line):
    result = line.split('-')[-1][:-1]
    if result == '0':
        return 2
    elif result == '1':
        return 1
    else:
        return 0


def process_game(args):
    line, cut_moves, cut_captures = args
    result = get_result(line)
    line = [l for l in line[:line.find(' {')].split(' ') if not '.' in l]
    byteboards = []
    board = chess.Board()

    for i, move in enumerate(line):
        if  i > cut_moves and not (cut_captures and 'x' in move):
            byteboards.append(get_byteboard(board, result))

        board.push_xboard(move)
    byteboards.append(get_byteboard(board, result))

    if len(byteboards) == 0:
        return [np.zeros((0, 33), dtype=np.uint8)]*3

    byteboards = np.stack(byteboards)
    output = [np.zeros((0, 33), dtype=np.uint8)]*3
    output[2-result] = byteboards
    return output


def game_gen(args):
    games = open(args[0])

    for line in games:
        if line[0] == '1':
            yield line, args[1], args[2]


def count_print(i, _time, val):
    if i % 1000 == 0:
        _time = time() - _time
        print('%d games loaded in %d seconds so far, averaging %.4f seconds per game...' % (i, _time, _time / i), end='\r')
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--out_id', type=str, default='', metavar='N',
                        help='unique id to allow multiple parsed datasets (default: "")')
    parser.add_argument('--test_percent', type=float, default=0.05, metavar='N',
                        help='Percentage of the data to devote to testing (default: 0.05)')
    parser.add_argument('--cut_moves', type=int, default=5, metavar='N',
                        help='Number of moves to cut off the beginning of each trajectory (default: 5)')
    parser.add_argument('--keep_captures', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='Number of workers for parsing games (default: 8)')
    args = parser.parse_args()
    args.cut_captures = not args.keep_captures
    if len(args.out_id) > 0:
        args.out_id = '_' + args.out_id

    print('Processing games, this may take a while...')
    _time = time()

    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), "w") as f:
        kwargs = {'maxshape': (None, 33), 'chunks': (1000, 33), 'dtype': 'uint8'}
        wins   = f.create_dataset('wins', (0, 33), **kwargs)
        losses = f.create_dataset('losses', (0, 33), **kwargs)
        ties   = f.create_dataset('ties', (0, 33), **kwargs)
        all_ds = [wins, losses, ties]

        with Pool(processes=8) as pool:
            for i, result in enumerate(pool.imap_unordered(process_game, game_gen(('data/{}/games.pgn'.format(args.dataset), args.cut_moves, args.cut_captures)))):
                count_print(i+1, _time, None)
                for i in range(3):
                    ds = all_ds[i]
                    res = result[i]
                    ds_len = len(ds)
                    ds.resize(size=ds_len+len(res), axis=0)
                    ds[ds_len:ds_len+len(res)] = res

    print('')
    print('Finished processing, saving...')
    batch_size = 1000
    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), rdcc_nbytes=1024**2*4000, rdcc_nslots=10**7) as f_in:
        with h5py.File('data/{}/byteboards{}.hdf5'.format(args.dataset, args.out_id), "w") as f_out:
            for group in ['train', 'test']:
                out_group = f_out.create_group(group)
                for dset in ['wins','losses','ties']:
                    games = f_in[dset]
                    num_test = int(args.test_percent * len(games))
                    num_train = len(games) - num_test
                    if group == 'train':
                        s = 0
                        e = num_train
                        num_here = num_train
                    else:
                        s = num_train
                        e = num_train + num_test
                        num_here = num_test

                    outset = out_group.create_dataset(dset, (num_here, 33), dtype='uint8')

                    num_batches = num_here // batch_size

                    for i in range(num_batches):
                        outset[batch_size*i:batch_size*(i+1)] = games[s+batch_size*i:s+batch_size*(i+1)]
                    outset[batch_size*num_batches:e-s] = games[s+batch_size*num_batches:e]

    os.remove('data/{}/temp.hdf5'.format(args.dataset))
    print('Done!')
