import numpy as np
import chess.pgn
import argparse
import h5py
from time import time
import ray
import ray_utils.queue_ray as q
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


def get_result(game):
    result = game.headers['Result']
    result = result.split('-')[0]
    if result == '1':
        return 2
    elif result == '0':
        return 1
    else:
        return 0


def process_game(board, result, moves):
    if len(moves) == 0:
        return [np.zeros((0, 33), dtype=np.uint8)]*3

    byteboards = []

    for move in moves:
        board.push(move)
        byteboards.append(get_byteboard(board, result))

    byteboards = np.stack(byteboards)
    output = [np.zeros((0, 33), dtype=np.uint8)]*3
    output[2-result] = byteboards
    return output


def game_gen(path):
    games = open(path)
    game = chess.pgn.read_game(games)
    while game is not None:
        yield game.board(), get_result(game), list(game.mainline_moves())
        game = chess.pgn.read_game(games)
    return


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
    ray.init()

    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    parser.add_argument('--test_percent', type=float, default=0.05, metavar='N',
                        help='Percentage of the data to devote to testing (default: 0.05)')
    parser.add_argument('--max_queue_size', type=int, default=0, metavar='N',
                        help='Max length of the parsing queue (default: 1000)')
    args = parser.parse_args()

    print('Processing games, this may take a while...')
    queue = q.Queue(maxsize=args.max_queue_size)
    q.put_queue.remote(game_gen, 'data/{}/games.pgn'.format(args.dataset), queue, process_game)

    #byteboards = [count_print(i+1, _time, process_game.remote(game)) for i, game in enumerate(game_gen(games))]
    #count_print(len(byteboards), _time, None)
    #print('')

    _time = time()

    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), "w") as f:
        train = f.create_group('train')
        tr_wins   = train.create_dataset('wins', (1000, 33), chunks=(1000, 33), dtype='uint8')
        tr_losses = train.create_dataset('losses', (1000, 33), chunks=(1000, 33), dtype='uint8')
        tr_ties   = train.create_dataset('ties', (1000, 33), chunks=(1000, 33), dtype='uint8')

        test = f.create_group('test')
        te_wins   = test.create_dataset('wins', (1000, 33), chunks=(1000, 33), dtype='uint8')
        te_losses = test.create_dataset('losses', (1000, 33), chunks=(1000, 33), dtype='uint8')
        te_ties   = test.create_dataset('ties', (1000, 33), chunks=(1000, 33), dtype='uint8')

        all_ds = [tr_wins, tr_losses, tr_ties, te_wins, te_losses, te_ties]

        for i, game in enumerate(q.get_queue(queue, 10)):
            count_print(i+1, _time, None)
            results = [a for result in game for a in split_data(result, args.test_percent)]
            for i in range(6):
                ds = all_ds[i]
                res = results[i]
                ds[len(ds):len(ds)+len(res)] = res

    print('Finished processing, saving...')
    batch_size = 1000
    with h5py.File('data/{}/temp.hdf5'.format(args.dataset), rdcc_nbytes=1024**2*4000, rdcc_nslots=10**7) as f_in:
        with h5py.File('data/{}/byteboards.hdf5'.format(args.dataset), "w") as f_out:
            for group in ['train', 'test']:
                out_group = f.create_group(group)
                for dset in ['wins','losses','ties']:
                    games = f_in['{}/{}'.format(group,dset)]
                    outset = out_group.create_dataset(dset, (len(games), 33), dtype='uint8')

                    num_batches = games.shape[0] // batch_size
                    inds        = [slice(batch_size*i, batch_size*(i+1)) for i in range(num_batches)] + [slice(batch_size*num_batches, len(games))]

                    for i, batch in enumerate(inds):
                        outset[batch] = games[batch]

    os.remove('data/{}/temp.hdf5'.format(args.dataset))
    print('Done!')
