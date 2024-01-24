import numpy as np
from tensorflow import keras

# Define the custom dataset for letter 'A'
A_data = np.array([
    [160, 77, 113, 142, 597, [-2.69, 7.86, -5.62, -2.69, 7.86, -5.62]],
    [151, 69, 100, 140, 590, [-2.70, 7.85, -5.63, -2.70, 7.85, -5.63]],
    [159, 75, 105, 141, 602, [-2.68, 7.87, -5.61, -2.68, 7.87, -5.61]],
    [154, 71, 112, 137, 595, [-2.71, 7.84, -5.64, -2.71, 7.84, -5.64]],
    [150, 72, 110, 144, 589, [-2.72, 7.83, -5.65, -2.72, 7.83, -5.65]]
])

# Define the custom dataset for letter 'B'
B_data = np.array([
    [120, 80, 100, 90, 400, [5.6, 6.7, -4.2, -5.1, -6.7, -5.5]],
    [115, 85, 95, 95, 395, [5.7, 6.8, -4.3, -5.2, -6.8, -5.6]],
    [118, 82, 98, 91, 398, [5.8, 6.9, -4.4, -5.3, -6.9, -5.7]],
    [122, 79, 101, 93, 402, [5.9, 7.0, -4.5, -5.4, -7.0, -5.8]],
    [116, 83, 97, 94, 396, [6.0, 7.1, -4.6, -5.5, -7.1, -5.9]]
])

# Combine the datasets and create labels
X_train = np.concatenate((A_data[:, :-1], B_data[:, :-1]))
y_train = np.concatenate((np.full((len(A_data),), 'A'), np.full((len(B_data),), 'B')))

# Shuffle the data
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

# Convert to float32
X_train = X_train.astype('float32')

# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=5)
