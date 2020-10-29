from models.autoencoder import AE
import numpy as np
import torch

model = AE()
state = torch.load('checkpoints/best_autoencoder.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
games = np.load('data/bitboards.npy')

batch_size  = 50
num_batches = games.shape[0] // batch_size
extras      = games[batch_size*num_batches:]

batched_games = np.split(games[:batch_size*num_batches], batch_size) + [extras]

def featurize(game):
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

feat_games = [featurize(batch) for batch in batched_games]
featurized = np.vstack(feat_games)

np.save('./data/features.npy', featurized)
