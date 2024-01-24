import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(11,))
print(inputs.dtype)
dense_1 = keras.layers.Dense(64, activation='relu')(inputs)
dense_2 = keras.layers.Dense(45, activation='relu')(dense_1)
outputs = keras.layers.Dense(26, activation='softmax')(dense_2)
model = keras.Model(inputs=inputs,outputs=outputs,name='SLTS')

