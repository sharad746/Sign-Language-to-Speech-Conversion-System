import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import os


inp = np.array([[160, 77, 113, 142, 597,-2.69, 7.86, -5.62, -2.69, 7.86, -5.62],
        [161, 79, 115, 144, 597, -2.28, 8.02, -5.69, -2.28, 8.02, -5.69],
        [162, 81, 117, 145, 598, -2.56, 7.88, -5.37, -2.56, 7.88, -5.37],
        [163, 82, 118, 146, 598, -2.83, 7.81, -5.77, -2.83, 7.81, -5.77],
        [163, 72, 120, 148, 598, -2.42, 7.88, -5.54, -2.42, 7.88, -5.54],
        [164, 84, 122, 148, 599, -2.86, 7.82, -5.66, -2.86, 7.82, -5.66],
        [164, 86, 124, 149, 599, -2.99, 7.93, -5.72, -2.99, 7.93, -5.72],
        [164, 88, 126, 149, 599, -2.89, 7.74, -5.73, -2.89, 7.74, -5.73],
        [165, 91, 127, 149, 599, -2.97, 7.76, -5.78, -2.97, 7.76, -5.78],
        [166, 94, 129, 150, 599, -3.02, 7.82, -5.65, -3.02, 7.82, -5.65],
        [165, 84, 125, 149, 599, -3.17, 7.82, -5.80, -3.17, 7.82, -5.80],
        [165, 98, 121, 138, 600, -2.86, 7.71, -5.81, -2.86, 7.71, -5.81]
        ])



alp = np.array([ 'A','A','A','A','A','A','A','A','A','A','A','A'])

(X_train, y_train) = (inp,alp)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)



model = keras.Sequential([

    keras.layers.Dense(1,  input_shape= (11,), activation='sigmoid')

])
#print(len(X_train))
#print(X_train.shape[0])
model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train_encoded, epochs=5)

