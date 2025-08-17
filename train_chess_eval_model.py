import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.utils import Sequence

# Data generator for chunked training

# Prepare chunk files
chunk_files = [f'chess_data_encoded_chunk{i}.npz' for i in range(20)]
train_files = chunk_files[:-2]  # Use last 2 for validation
val_file = chunk_files[-1]

# Validation data
with np.load(val_file) as val_data:
    X_val, Y_val = val_data['X'], val_data['Y']

# tf.data.Dataset for training
def chunked_data_generator(chunk_files, batch_size=1024, batches_per_chunk=100, shuffle_chunks=True):
    while True:
        files = chunk_files.copy()
        if shuffle_chunks:
            np.random.shuffle(files)
        for chunk_file in files:
            print(f"Loading chunk file: {chunk_file}")
            data = np.load(chunk_file)
            X, Y = data['X'], data['Y']
            for _ in range(batches_per_chunk):
                idxs = np.random.choice(len(X), batch_size, replace=False)
                yield X[idxs], Y[idxs]

batch_size = 1024
batches_per_chunk = 100
train_dataset = tf.data.Dataset.from_generator(
    lambda: chunked_data_generator(train_files, batch_size, batches_per_chunk, shuffle_chunks=True),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 8, 8, 18), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    )
)


# Model definition
model = models.Sequential([
    layers.Input(shape=(8, 8, 18)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
progress = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}"))

# Training
steps_per_epoch = batches_per_chunk * len(train_files)
history = model.fit(
    train_dataset,
    validation_data=(X_val, Y_val),
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stop, progress],
    verbose=1
)

# Save loss data for later analysis
import pandas as pd
loss_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})
loss_df.to_csv('training_loss_data.csv', index=False)

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

# Save model
model.save('chess_eval_model.keras')
print('Model saved as chess_eval_model.keras')
