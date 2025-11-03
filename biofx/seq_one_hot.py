import numpy as np
import tensorflow as tf
from tensorflow import keras

# Example one-hot encoding function for a DNA sequence
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 
               'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping[base] for base in sequence])

# Example DNA sequences
sequences = ["ATCG", "GCTA", "TACG"]
encoded_sequences = np.array([one_hot_encode(seq) for seq in sequences])

# Define a simple CNN model
model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(4, 4)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Note: Adjust input_shape and loss function based on your problem

# Dummy labels for illustration
labels = np.array([0, 1, 0])

# Train the model
model.fit(encoded_sequences, labels, epochs=10)
