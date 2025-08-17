import glob
import numpy as np

output_npz = "chess_data_encoded_all.npz"

#chunk_files = sorted(glob.glob("chess_data_encoded_chunk*.npz"))[10:]
chunk_files = ["chess_data_encoded.npz", "chess_data_encoded2.npz"]
if chunk_files:
    print(f"Merging {len(chunk_files)} chunk files into {output_npz}...")
    X_list = []
    Y_list = []
    for f in chunk_files:
        data = np.load(f)
        X_list.append(data['X'])
        Y_list.append(data['Y'])
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    np.savez_compressed(output_npz, X=X_all, Y=Y_all)
    print(f"Merged file saved as {output_npz} with {X_all.shape[0]} samples.")