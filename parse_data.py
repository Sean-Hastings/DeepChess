import numpy as np
import chess
import argparse
import h5py
from time import time
import os
import re
from multiprocessing.pool import Pool


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
    result = line[line.rfind('-')+1:-1]
    if result == '0':
        return 2
    elif result == '1':
        return 1
    else:
        return 0


def process_game(args):
    line, cut_moves, cut_captures = args

    byteboards = []
    result = get_result(line)
    board  = chess.Board()

    moves = re.split(' |[^\s]*\.|{.*}|0-1|1-0|1\/2-1\/2', line)
    moves = [l for l in moves if len(l) > 0]

    for i, move in enumerate(moves):
        if i >= cut_moves and not (cut_captures and 'x' in move):
            byteboards.append(get_byteboard(board, result))
        board.push_xboard(move)

    if len(byteboards) == 0:
        return [np.zeros((0, 33), dtype=np.uint8)]*3

    byteboards = np.stack(byteboards)
    output = [np.zeros((0, 33), dtype=np.uint8)]*3
    output[2-result] = byteboards
    return output



def game_gen(args):
    lines = open(args[0])
    cur_line = ''
    for line in lines:
        if line[0] not in '[\n':
            cur_line += line[:-1] + ' '
        elif len(cur_line) > 0:
            yield cur_line, args[1], args[2]
            cur_line = ''


def count_print(i, _time, val):
    if i % 1000 == 0:
        _time = time() - _time
        print('%d games loaded in %d seconds so far, averaging %.4f seconds per game...' % (i, _time, _time / i), end='\r')
    return val


def split_data(data, test_percent):
    half_test = int((len(data)*test_percent)//2)
    test_data = data[:half_test]
    data      = data[half_test:]
    p         = np.random.permutation(len(data))
    test_data = np.concatenate([test_data, data[p[:half_test]]], axis=0)
    data      = data[p[half_test:]]
    return (data, test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--id', type=str, default='', metavar='N',
                        help='unique identifier to allow for multiple parsings of the same dataset to be stored (default: "")')
    parser.add_argument('--test_percent', type=float, default=0.05, metavar='N',
                        help='Percentage of the data to devote to testing (default: 0.05)')
    parser.add_argument('--cut_moves', type=int, default=5, metavar='N',
                        help='Number of moves to cut off the beginning of each trajectory (default: 5)')
    parser.add_argument('--keep_captures', action='store_true', default=False,
                        help='Flag to not cut out capturing moves')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='Number of game-parsing processes (default: 8)')
    args = parser.parse_args()
    args.cut_captures = not args.keep_captures
    if len(args.id):
        args.id = '_' + args.id

    print('Processing games, this may take a while...')

    _time = time()

    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), "w") as f:
        kwargs = {'maxshape': (None, 33), 'chunks': (1000, 33), 'dtype': 'uint8'}
        wins   = f.create_dataset('wins', (0, 33), **kwargs)
        losses = f.create_dataset('losses', (0, 33), **kwargs)
        ties   = f.create_dataset('ties', (0, 33), **kwargs)

        all_ds = [wins, losses, ties]

        with Pool(args.num_workers) as pool:
            gen = game_gen(('data/{}/games.pgn'.format(args.dataset), args.cut_moves, args.cut_captures))
            for i, game in enumerate(pool.imap_unordered(process_game, gen, chunksize=1000)):
                count_print(i+1, _time, None)
                for i in range(3):
                    ds = all_ds[i]
                    res = game[i]
                    lends = len(ds)
                    ds.resize(lends+len(res), axis=0)
                    ds[lends:lends+len(res)] = res

    print('')
    print('Finished processing, saving...')
    raise Exception()
    batch_size = 1000
    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), rdcc_nbytes=1024**2*4000, rdcc_nslots=10**7) as f_in:
        with h5py.File('data/{}/byteboards{}.hdf5'.format(args.dataset, args.id), "w") as f_out:
            for group in ['train', 'test']:
                out_group = f_out.create_group(group)
                for dset in ['wins','losses','ties']:
                    games = f_in[dset]
                    outset = out_group.create_dataset(dset, (len(games), 33), dtype='uint8')

                    num_batches = games.shape[0] // batch_size
                    inds        = [slice(batch_size*i, batch_size*(i+1)) for i in range(num_batches)] + [slice(batch_size*num_batches, len(games))]

                    for i, batch in enumerate(inds):
                        outset[batch] = games[batch]

    os.remove('data/{}/temp.hdf5'.format(args.dataset))
    print('Done!')
