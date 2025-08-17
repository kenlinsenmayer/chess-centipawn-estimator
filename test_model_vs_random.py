import numpy as np
import tensorflow as tf
import chess
import random
from tqdm import trange

# Load trained model
model = tf.keras.models.load_model('chess_eval_model.keras')

def encode_board(board):
    # Use the actual encoding function from fen_encoder.py
    from fen_encoder import fen_to_tensor
    return fen_to_tensor(board.fen())

def pick_best_move(board, model):
    legal_moves = list(board.legal_moves)
    encoded_boards = []
    for move in legal_moves:
        board.push(move)
        encoded_boards.append(encode_board(board))
        board.pop()
    x = np.stack(encoded_boards, axis=0)  # shape: (num_moves, 8, 8, 18)
    scores = model.predict(x, verbose=0).flatten()
    # For white, higher is better
    best_idx = np.argmax(scores)
    return legal_moves[best_idx]

def play_game(model):
    board = chess.Board()
    move_count = 0
    max_moves = 150
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            move = pick_best_move(board, model)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        move_count += 1
    if move_count >= max_moves and not board.is_game_over():
        result = '1/2-1/2'  # Draw if max move limit reached
    else:
        result = board.result()
    return result, move_count

def main():
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    for i in trange(100, desc='Games'):
        result, move_count = play_game(model)
        results[result] += 1
        print(f"Game {i+1}: Result={result}, Moves={move_count}")
    print('Results over 100 games:')
    print(f"Agent wins:   {results['1-0']}")
    print(f"Random wins:  {results['0-1']}")
    print(f"Draws:        {results['1/2-1/2']}")

if __name__ == '__main__':
    main()
