import numpy as np

val_file = 'chess_data_encoded_chunk19.npz'

data = np.load(val_file)
X = data['X']
Y = data['Y']
print('X shape:', X.shape)
print('Y shape:', Y.shape)
