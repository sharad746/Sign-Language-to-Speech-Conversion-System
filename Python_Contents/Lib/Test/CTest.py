import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder



#c_y_Test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/C_data/C_y_test.txt", dtype='str')

B_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_train.txt")
B_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_train.txt", dtype='str')
B_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_test.txt")
B_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_test.txt", dtype='str')
A_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_train.txt")
A_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_train.txt", dtype='str')
A_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_test.txt")
A_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_test.txt", dtype='str')
(X_train, y_train), (X_test, y_test) =(np.concatenate((B_X_train,A_X_train)), np.concatenate((B_y_train,A_y_train))), (np.concatenate((B_X_test, A_X_test)), np.concatenate((B_y_test,A_y_test)))

labelEncoder = LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(y_train)
y_test_encoded = labelEncoder.fit_transform(y_test)
print(y_test)
print(y_test_encoded)
print(y_train)
print(y_train_encoded)

predict = np.array([[138, 127, 196, 218, 607, 1.53, 8.83, -4.91, -0.10, -0.01, 0.01],
    [157, 98, 136, 134, 603, -0.34, 9.68, -1.16, -0.11, -0.03, 0.01],
                    [138, 127, 197, 218, 607, 1.64, 8.84, -4.90, -0.11, -0.01, 0.02],


                    [156, 98, 136, 135, 602, -0.33, 9.71, -1.09, -0.12, 0.00, 0.01]])

#building the RNN GRU model
input_layer = keras.Input(shape=(11,))
dense_1 = keras.layers.Dense(11, activation='relu')(input_layer)
reshape_1 = keras.layers.Reshape((1, 11))(dense_1)
dense_2 = keras.layers.GRU(32, activation='tanh')(reshape_1)
dense_3 = keras.layers.Dense(24,activation='tanh')(dense_2)
output_layer = keras.layers.Dense(1, activation='sigmoid')(dense_3)
model = keras.Model(inputs=input_layer,outputs=output_layer,name='SLTS')

#compiling the model
model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['Precision']
)

#trains the model
model.fit(X_train, y_train_encoded, epochs=50)
#model.save('SLTS_model.h5')
predicted_labels = labelEncoder.inverse_transform(model.predict(X_test).astype(int).flatten())
#predicted_labels = model.predict(X_test)
print(predicted_labels)