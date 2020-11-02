from models.autoencoder import AE
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE().to(device)
state = torch.load('checkpoints/best_autoencoder.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
games = np.load('data/byteboards.npy')

batch_size  = 4096
num_batches = games.shape[0] // batch_size
extras      = games[batch_size*num_batches:]

batched_games = np.split(games[:batch_size*num_batches], num_batches) + [extras]

def featurize(game, index, max_indices):
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor).to(device))
    print('{} / {}'.format(index, max_indices), end='\r')
    return enc.cpu().detach().numpy()

feat_games = [featurize(batch, i+1, len(batched_games)) for i, batch in enumerate(batched_games)]
featurized = np.vstack(feat_games)

np.save('./data/features.npy', featurized)
