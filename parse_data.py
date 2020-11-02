import numpy as np
import chess.pgn
import argparse
from time import time
import ray
ray.init()


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


@ray.remote
def process_game(game):
    placeholder_board = np.zeros(33).astype(np.uint8)
    placeholder_label = 0

    byteboards = []
    result = get_result(game)

    board = game.board()

    for move in game.mainline_moves():
        board.push(move)
        byteboard = get_byteboard(board, result)
        byteboards.append(byteboard)

    byteboards.append(placeholder_board)

    result = np.stack(byteboards)
    return result


def game_gen(games):
    game = chess.pgn.read_game(games)
    while game is not None:
        yield game
        game = chess.pgn.read_game(games)
    return


def count_print(i, _time, val):
    if i % 1000 == 0:
        _time = time() - _time
        print('%d games loaded in %d seconds so far, averaging %.4f seconds per game...' % (i, _time, _time / i), end='\r')
    return val


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(25000)

    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--dataset', type=str, default='ccrl', metavar='N',
                        help='name of the dataset to parse (default: ccrl)')
    args = parser.parse_args()

    games = open('data/{}.pgn'.format(args.dataset))
    _time = time()

    print('Processing games, this may take a while...')
    byteboards = [count_print(i+1, _time, process_game.remote(game)) for i, game in enumerate(game_gen(games))]
    count_print(len(byteboards), _time, None)
    print('')
    byteboards = np.concatenate([ray.get(game) for game in byteboards], axis=0)

    print('byteboards shape:', byteboards.shape)
    print('byteboards size in bytes:', byteboards.nbytes)
    print('Finished processing, saving...')

    np.save('./data/{}_byteboards.npy'.format(args.dataset), byteboards)

    print('Done!')
