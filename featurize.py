from models.autoencoder import AE
from utils import bitboard_from_byteboard
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE().to(device)
state = torch.load('checkpoints/autoencoder/overnight/best.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
games = np.load('data/ccrl_byteboards.npy')

batch_size  = 4096
num_batches = games.shape[0] // batch_size
extras      = games[batch_size*num_batches:]

batched_games = np.split(games[:batch_size*num_batches], num_batches) + [extras]

def featurize(game, index, max_indices):
    game, _ = bitboard_from_byteboard(game)
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor).to(device))
    print('{} / {}'.format(index, max_indices), end='\r')
    return enc.cpu().detach().numpy()

feat_games = [featurize(batch, i+1, len(batched_games)) for i, batch in enumerate(batched_games)]
featurized = np.vstack(feat_games)

np.save('./data/ccrl_features.npy', featurized)
