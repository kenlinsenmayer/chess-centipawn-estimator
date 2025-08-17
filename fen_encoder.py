import numpy as np
import pandas as pd
import chess
import chess.pgn
import chess.engine
import os
from tqdm import tqdm

# Mapping from piece symbol to channel index
PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

CASTLING_INDICES = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}


def fen_to_tensor(fen):
    """
    Convert a FEN string to an (8, 8, 18) numpy array encoding.
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 18), dtype=np.float32)

    # 1. Piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            channel = PIECE_TO_CHANNEL[piece.symbol()]
            tensor[row, col, channel] = 1.0

    # 2. Side to move
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # 3. Castling rights
    for flag, idx in CASTLING_INDICES.items():
        if flag in board.castling_xfen():
            tensor[:, :, 13 + idx] = 1.0
        else:
            tensor[:, :, 13 + idx] = 0.0

    # 4. En passant
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[row, col, 17] = 1.0

    return tensor


def process_csv(input_csv, output_npz, fen_col='fen', target_col='cp_standardized_clipped', max_rows=None):
    """
    Process the CSV in manageable chunks and save X (encodings) and Y (targets) as .npz files.
    """
    chunk_size = 1000000  # Adjust as needed for your memory
    chunk_iter = pd.read_csv(input_csv, usecols=[fen_col, target_col], chunksize=chunk_size, nrows=max_rows)
    chunk_idx = 0
    total = 0
    for df in chunk_iter:
        X = []
        Y = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunk {chunk_idx}"):
            try:
                tensor = fen_to_tensor(row[fen_col])
                X.append(tensor)
                Y.append(row[target_col])
            except Exception:
                continue  # skip malformed FENs or missing data
        if X:
            X = np.stack(X)
            Y = np.array(Y, dtype=np.float32)
            out_file = output_npz.replace('.npz', f'_chunk{chunk_idx}.npz')
            np.savez_compressed(out_file, X=X, Y=Y)
            print(f"Saved {len(X)} samples to {out_file}")
            total += len(X)
        chunk_idx += 1
    print(f"Total samples saved: {total}")


if __name__ == "__main__":
    # Change these as needed
    input_csv = "lichess_fen_cp_20M.csv"
    output_npz = "chess_data_encoded.npz"
    process_csv(input_csv, output_npz)

    # Merge all chunk files into one
    import glob
    chunk_files = sorted(glob.glob(output_npz.replace('.npz', '_chunk*.npz')))
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
