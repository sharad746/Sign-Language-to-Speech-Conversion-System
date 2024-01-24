import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pyttsx3
from sklearn.preprocessing import StandardScaler

#A Train and test dataset
X_train_dataset = np.genfromtxt("C:/Users/rikes/OneDrive/Documents/AB_Train.csv", skip_header=1,usecols=range(11), delimiter=',')
y_train_dataset_A = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/txt.txt", dtype='str')
y_train_dataset_B = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/txt2.txt", dtype='str')
X_test_dataset = np.genfromtxt("C:/Users/rikes/OneDrive/Documents/AB_Test2.csv", skip_header=1,usecols=range(11), delimiter=',')
y_test_dataset_A = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/testA.txt", dtype='str')
y_test_dataset_B = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/testB.txt", dtype='str')

# create training and testing data
X = X_train_dataset[ : , 0:11]
y = np.concatenate((y_train_dataset_A,y_train_dataset_B))
test_X = X_test_dataset[ : , 0:11]
test_y = np.concatenate((y_test_dataset_A,y_test_dataset_B))
X_train, y_train, X_test, y_test = (X, y, test_X, test_y)
print(y_train)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

labelEncoder = LabelEncoder() #instantiating the Label Encoder
y_train_encoded = labelEncoder.fit_transform(y_train)

y_test_encoded = labelEncoder.fit_transform(y_test)

print(y_train_encoded[697:704])

predict = np.array([
            [125, 135, 187, 229, 609, 0.20, 9.16, -3.93, -0.09, -0.07, 0.08],
            [158, 89, 132, 150, 600, -0.48, 9.38, -1.99, -0.03, 0.06, -0.01],
            [159, 89, 132, 150, 601, -0.55, 9.61, -1.93, -0.07, 0.00, 0.00],
            [125, 135, 188, 229, 610, 0.51, 9.24, -3.80, -0.09, -0.00, 0.00],
            [125, 135, 188, 228, 609, 0.33, 9.29, -3.93, -0.09, -0.01, 0.04]
                   ])
#predict = sc.fit_transform(predict)

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
    metrics=['accuracy']
)

#trains the model
model.fit(X_train, y_train_encoded, epochs=50)
#model.save('SLTS_model.h5')


predicted_labels = labelEncoder.inverse_transform(model.predict(predict).astype('int').flatten())
#predicted_labels_reshaped = predicted_labels.reshape(5,11)

#predicted_labels = model.predict(X_test)
print(predicted_labels)

engine = pyttsx3.init()

# Set properties for the speech
engine.setProperty('rate', 100)  # Set the speaking rate (words per minute)
engine.setProperty('volume', 1)  # Set the volume (float between 0 and 1)


# Convert text to speech

engine.say(predicted_labels)
engine.runAndWait()