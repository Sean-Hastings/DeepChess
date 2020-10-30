import numpy as np
import chess.pgn

def get_bitboard(board):
    '''
    params
    ------

    board : chess.pgn board object
        board to get state from

    returns
    -------

    bitboard representation of the state of the game
    64 * 6 + 5 dim binary numpy vector
    64 squares, 6 pieces, '1' indicates the piece is at a square
    5 extra dimensions for castling rights queenside/kingside and whose turn

    '''

    bitboard = np.zeros(64*6*2+5)

    piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

    for i in range(64):
        if board.piece_at(i):
            color = int(board.piece_at(i).color) + 1
            bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard

def get_result(game):
    result = game.headers['Result']
    result = result.split('-')
    if result[0] == '1':
        return 1
    elif result[0] == '0':
        return -1
    else:
        return 0

games = open('data/games.pgn')
game = chess.pgn.read_game(games)
bitboards = []
labels = []
num_games = 0

placeholder_board = np.zeros(773)
placeholder_label = 0

while game is not None:
    if num_games > 0 and num_games % 1000 == 0:
        print('# Games: %d' % num_games)
        print('# Moves: %d' % len(bitboards))
        print('Avg. Moves / Game: %.1f' % (len(bitboards) / num_games))
        print('=======================')

    num_games += 1

    result = get_result(game)

    board = game.board()

    i = 0
    for move in game.mainline_moves():
        board.push(move)
        bitboard = get_bitboard(board)
        if i > 50:
            bitboards.append(bitboard)
            labels.append(result)
        i += 1

    bitboards.append(placeholder_board)
    labels.append(placeholder_label)

    game = chess.pgn.read_game(games)

bitboards = np.array(bitboards)
labels = np.array(labels)

print('bitboards shape:', bitboards.shape)
print('labels shape:', labels.shape)
print('Finished processing, saving...')

np.save('./data/bitboards.npy', bitboards)
np.save('./data/labels.npy', labels)

print('Done!')
