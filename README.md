# Chess Evaluation Neural Network Project Summary

## Project Overview
This project aims to train a neural network to evaluate chess positions using a large dataset of FEN strings and centipawn values. The model is trained to predict the value of a board position from White's perspective, enabling it to play chess by selecting moves that maximize its evaluation.

---

## Workspace File Descriptions

- **fen_encoder.py**: Contains functions to encode FEN strings into (8, 8, 18) numpy arrays suitable for neural network input. Also includes utilities for processing large CSV datasets into chunked .npz files for efficient training.
- **generate_data.py**: (If present) Used for generating or preparing additional data for training or testing.
- **analyze_csv.py, inspect_entries.py, subset_csv.py, clip_standardized_cp.py, standardize_cp.py**: Utility scripts for analyzing, inspecting, subsetting, and standardizing the chess dataset.
- **lichess_fen_cp_20M.csv**: The original large dataset of FEN strings and centipawn values.
- **chess_data_encoded_chunk*.npz**: Chunked, preprocessed data files containing encoded board positions and target values for training.
- **train_chess_eval_model.py**: The main training script. Loads chunked data, builds and trains the CNN model, and saves the trained model and training loss plots/data.
- **training_loss.png**: Plot of training and validation loss over epochs.
- **training_loss_data.csv**: CSV file containing the loss and validation loss per epoch for further analysis.
- **chess_eval_model.keras**: The saved trained model.
- **test_model_vs_random.py**: Script to test the trained model by playing as White against a random-move opponent, reporting results over multiple games.

---

## Model Training
- **Data Pipeline**: The large CSV was processed into chunked .npz files using `fen_encoder.py` to avoid memory issues. Each chunk contains encoded FENs and their corresponding centipawn values.
- **Model Architecture**: A convolutional neural network (CNN) was used, taking (8, 8, 18) tensors as input and outputting a single value (board evaluation for White).
- **Training**: The model was trained using TensorFlow/Keras, with chunked data loaded sequentially using `tf.data.Dataset.from_generator` for memory efficiency. Early stopping and validation split were used to prevent overfitting. Training and validation loss were plotted and saved.

---

## Results So Far
- The model was tested as White against a random-move opponent for 100 games.
- **Results:**
  - Agent (White) wins: 25
  - Random wins: 0
  - Draws: 75
- Most games ended in draws due to the 150-move cap, but the agent consistently outperformed random play, demonstrating that the model learned useful evaluation features.

---

## Recommendations for Future Improvements
- **Improve Model Strength**: Train on more data, use deeper or more advanced architectures, or fine-tune hyperparameters.
- **Play as Black**: Extend the agent to play as Black and/or against stronger opponents (e.g., simple heuristics or other engines).
- **Self-Play**: Implement self-play to further improve the model.
- **Endgame Handling**: Add special handling or training for endgames to reduce the number of drawn games due to the move cap.
- **Evaluation Metrics**: Track additional metrics such as average centipawn loss, blunder rate, or compare against known engine evaluations.
- **User Interface**: Build a simple UI or web app to play against the trained model interactively.

---

---

## Recent Improvements
- **Batch Move Evaluation Implemented**: The `pick_best_move` function now batch-evaluates all legal moves at once, leveraging model parallelism for much faster move selection and overall gameplay/testing speed.

## Conclusion
This project successfully demonstrates a pipeline for training and evaluating a neural network to assess chess positions. The model shows clear improvement over random play, and the codebase is well-structured for further research and development.
