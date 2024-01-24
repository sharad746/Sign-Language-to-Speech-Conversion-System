import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pyttsx3
from sklearn.preprocessing import StandardScaler

#A Train and test dataset
A_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_train.txt")
print(A_X_train)
A_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_train.txt", dtype='str')
A_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_test.txt")
A_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_test.txt", dtype='str')


#B train and test dataset
B_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_train.txt")
B_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_train.txt", dtype='str')
B_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_test.txt")
B_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_test.txt", dtype='str')


# concatening and storing np arrays in train and test variables
(X_train, y_train),(X_test, y_test) = (np.concatenate((A_X_train,B_X_train)), np.concatenate((A_y_train,B_y_train))), (np.concatenate((A_X_test, B_X_test)), np.concatenate((A_y_test,B_y_test)))


labelEncoder = LabelEncoder() #instantiating the Label Encoder
y_train_encoded = labelEncoder.fit_transform(y_train)
y_test_encoded = labelEncoder.fit_transform(y_test)
print(X_test)
print(y_test)

predict = np.array([[125, 135, 187, 229, 609, 0.20, 9.16, -3.93, -0.09, -0.07, 0.08],
[158, 89, 132, 150, 600, -0.48, 9.38, -1.99, -0.03, 0.06, -0.01],
[159, 89, 132, 150, 601, -0.55, 9.61, -1.93, -0.07, 0.00, 0.00],
[125, 135, 188, 229, 610, 0.51, 9.24, -3.80, -0.09, -0.00, 0.00],
[125, 135, 188, 228, 609, 0.33, 9.29, -3.93, -0.09, -0.01, 0.04]
                   ])

#building the RNN GRU model
input_layer = keras.Input(shape=(11,))
dense_1 = keras.layers.Dense(11, activation='tanh')(input_layer)
reshape_1 = keras.layers.Reshape((1, 11))(dense_1)
gru_layer = keras.layers.GRU(64)(reshape_1)
dense_3 = keras.layers.Dense(32,activation='tanh')(gru_layer)
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

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')

predicted_labels = labelEncoder.inverse_transform(model.predict(predict).astype('int').flatten())
#predicted_labels = model.predict(X_test)
print(predicted_labels)

engine = pyttsx3.init()

# Set properties for the speech
engine.setProperty('rate', 100)  # Set the speaking rate (words per minute)
engine.setProperty('volume', 1)  # Set the volume (float between 0 and 1)


# Convert text to speech

engine.say(predicted_labels)
engine.runAndWait()