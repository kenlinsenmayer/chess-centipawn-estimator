import numpy as np

# Input files
file1 = "chess_data_encoded.npz"
file2 = "chess_data_encoded2.npz"
output_file = "chess_data_encoded_all.npz"

# Open both files as memory-mapped arrays
with np.load(file1, mmap_mode='r') as data1, np.load(file2, mmap_mode='r') as data2:
    X1, Y1 = data1['X'], data1['Y']
    X2, Y2 = data2['X'], data2['Y']

    # Get shapes
    n1, n2 = X1.shape[0], X2.shape[0]
    X_shape = (n1 + n2,) + X1.shape[1:]
    Y_shape = (n1 + n2,)

    # Create memmap files for output
    X_out = np.lib.format.open_memmap('X_tmp.npy', mode='w+', dtype=X1.dtype, shape=X_shape)
    Y_out = np.lib.format.open_memmap('Y_tmp.npy', mode='w+', dtype=Y1.dtype, shape=Y_shape)

    # Copy data in chunks
    chunk_size = 100000
    for start in range(0, n1, chunk_size):
        end = min(start + chunk_size, n1)
        X_out[start:end] = X1[start:end]
        Y_out[start:end] = Y1[start:end]
    for start in range(0, n2, chunk_size):
        end = min(start + chunk_size, n2)
        X_out[n1+start:n1+end] = X2[start:end]
        Y_out[n1+start:n1+end] = Y2[start:end]

    # Save to compressed npz
    np.savez_compressed(output_file, X=X_out, Y=Y_out)

# Clean up temporary files
import os
os.remove('X_tmp.npy')
os.remove('Y_tmp.npy')

print(f"Combined file saved as {output_file}")
