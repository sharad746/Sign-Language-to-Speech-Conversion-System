import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder

a = np.array([[180, 98, 146, 171, 609, 0.57, 2.44, -0.46, -0.02, -0.05, 0.00],
            [180, 97, 149, 172, 610, 2.09, 9.21, -3.59, 0.04, -0.20, 0.05],
            [182, 96, 148, 172, 609, 1.85, 9.29, -3.05, -0.16, -0.10, 0.05],
            [180, 96, 147, 171, 610, 1.71, 9.29, -3.23, -0.11, -0.00, -0.01],
            [181, 96, 147, 172, 610, 1.88, 9.29, -2.70, -0.09, 0.05, -0.01],
            [180, 97, 147, 172, 609, 1.79, 9.30, -2.77, -0.20, -0.02, 0.11],
            [181, 97, 148, 171, 610, 1.82, 9.36, -3.29, -0.04, -0.05, -0.03],
            [181, 97, 148, 172, 610, 1.64, 9.36, -2.83, -0.05, -0.02, -0.05],
            [182, 98, 148, 172, 610, 1.94, 9.31, -3.28, -0.10, 0.05, 0.03],
            [180, 96, 147, 170, 609, 1.70, 9.26, -3.11, -0.03, 0.08, 0.01],
            [181, 96, 148, 172, 611, 2.20, 9.28, -2.79, -0.11, 0.05, 0.05],
            [181, 97, 149, 171, 610, 1.85, 9.52, -3.39, -0.01, 0.05, -0.03],
            [181, 96, 147, 172, 609, 1.98, 9.42, -3.51, -0.04, 0.22, -0.00],
            [180, 96, 148, 172, 609, 1.51, 9.45, -3.43, -0.02, 0.17, -0.06],
            [181, 98, 148, 172, 610, 1.83, 9.41, -3.43, -0.08, 0.09, -0.01],
            [180, 96, 148, 171, 611, 1.45, 9.33, -3.34, -0.02, 0.07, -0.04],
            [180, 96, 148, 171, 611, 2.11, 9.57, -2.50, -0.17, -0.28, 0.09],
            [182, 97, 148, 171, 610, 1.44, 9.09, -3.56, -0.03, -0.04, -0.03],
            [181, 97, 148, 171, 610, 1.54, 9.47, -3.08, -0.11, 0.03, 0.05],
            [181, 97, 148, 171, 610, 1.93, 9.45, -3.32, 0.01, 0.17, -0.01],
            [181, 96, 149, 171, 609, 1.97, 9.48, -3.39, -0.05, 0.18, -0.01],
            [181, 97, 148, 171, 609, 1.41, 9.57, -3.06, -0.13, -0.05, 0.00],
            [181, 96, 148, 172, 611, 1.57, 9.05, -3.58, 0.02, -0.06, 0.05],
            [181, 97, 148, 173, 610, 1.70, 9.42, -2.97, -0.13, -0.02, 0.02],
            [181, 96, 148, 171, 611, 1.50, 9.31, -3.67, -0.12, -0.05, -0.02],
            [180, 97, 148, 172, 610, 1.67, 9.50, -3.25, -0.12, -0.12, 0.07],
            [181, 97, 149, 172, 610, 1.81, 9.44, -2.96, -0.04, -0.04, -0.01],
            [181, 97, 147, 171, 611, 1.53, 9.22, -3.29, -0.08, 0.01, 0.01],
            [182, 97, 147, 172, 610, 1.76, 9.33, -3.62, -0.08, -0.05, 0.04],
            [182, 97, 148, 173, 611, 1.74, 9.46, -3.32, -0.09, 0.04, 0.01],
            [181, 97, 147, 172, 610, 1.71, 9.45, -3.47, -0.07, -0.03, 0.00],
            [181, 97, 146, 171, 610, 1.75, 9.39, -3.07, -0.11, -0.03, 0.02],
            [181, 97, 148, 172, 610, 1.54, 9.24, -3.26, -0.13, -0.05, 0.03],
            [181, 97, 148, 172, 611, 1.75, 9.33, -3.31, -0.13, -0.06, 0.03],
            [180, 97, 147, 172, 611, 1.84, 9.36, -3.13, -0.09, -0.03, 0.01],
            [182, 97, 147, 170, 611, 1.65, 9.28, -3.58, -0.09, -0.01, 0.02],
            [182, 96, 149, 173, 609, 1.74, 9.42, -3.12, -0.14, -0.03, 0.04],
            [181, 97, 148, 171, 610, 1.66, 9.33, -3.39, -0.09, -0.02, 0.03],
            [181, 96, 148, 172, 610, 1.58, 9.50, -3.25, -0.14, 0.00, 0.01],
            [181, 97, 147, 171, 609, 1.62, 9.33, -3.35, -0.11, -0.10, 0.04],
            [182, 97, 148, 171, 611, 1.81, 9.27, -3.29, -0.15, -0.04, 0.04],
            [181, 97, 147, 172, 610, 1.76, 9.34, -3.15, -0.10, -0.03, 0.02],
            [181, 97, 148, 172, 610, 1.72, 9.18, -3.30, -0.14, -0.01, 0.01],
            [181, 97, 148, 172, 610, 2.04, 9.27, -3.14, -0.07, -0.02, 0.03],
            [181, 96, 148, 172, 610, 1.58, 9.34, -3.15, -0.09, 0.01, 0.03],
            [181, 97, 147, 171, 610, 2.11, 9.33, -3.17, -0.09, 0.01, 0.03],
            [181, 96, 148, 171, 611, 1.58, 9.41, -3.49, -0.08, 0.00, -0.01],
            [181, 97, 148, 172, 610, 1.63, 9.38, -3.37, -0.07, -0.02, 0.01],
            [181, 96, 148, 171, 610, 1.82, 9.40, -3.18, -0.09, -0.01, 0.04],
            [182, 97, 148, 172, 610, 1.80, 9.37, -3.00, -0.08, -0.01, 0.03]])

print(a.shape)

a1 = np.array(['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A',])
print(a1.shape)

b = np.array([[132, 127, 195, 214, 606, 0.58, 2.24, -1.08, -0.03, -0.00, 0.00],
                [132, 127, 194, 215, 607, 2.26, 8.97, -4.28, -0.13, -0.01, 0.01],
                [132, 127, 194, 215, 607, 2.32, 8.99, -4.24, -0.10, 0.00, 0.01],
                [133, 127, 194, 214, 606, 2.24, 8.95, -4.31, -0.10, -0.01, 0.01],
                [132, 127, 194, 214, 606, 2.34, 8.98, -4.12, -0.10, -0.01, 0.02],
                [132, 126, 194, 214, 607, 2.30, 8.97, -4.19, -0.12, -0.02, 0.02],
                [132, 127, 194, 215, 606, 2.26, 9.05, -4.11, -0.11, -0.01, 0.03],
                [132, 126, 194, 214, 606, 2.29, 9.03, -4.17, -0.11, -0.01, 0.01],
                [132, 127, 195, 215, 607, 2.41, 8.96, -4.30, -0.10, -0.02, 0.02],
                [132, 126, 195, 215, 607, 2.36, 8.98, -4.23, -0.09, -0.02, 0.02],
                [133, 127, 194, 214, 607, 2.26, 8.89, -4.30, -0.10, -0.01, 0.01],
                [132, 127, 195, 215, 607, 1.70, 8.88, -4.19, -0.15, -0.03, 0.08],
                [133, 126, 194, 215, 607, 2.33, 8.94, -4.12, -0.11, 0.01, 0.03],
                [132, 126, 194, 214, 606, 2.40, 8.94, -4.21, -0.11, -0.02, 0.01],
                [132, 127, 194, 215, 607, 2.35, 8.98, -4.32, -0.10, -0.01, -0.01],
                [133, 127, 195, 215, 606, 2.42, 8.91, -4.35, -0.08, -0.01, 0.02],
                [133, 127, 194, 215, 606, 2.35, 9.21, -4.17, -0.11, -0.12, 0.08],
                [133, 127, 195, 215, 606, 2.59, 8.92, -4.32, -0.04, -0.06, 0.02],
                [133, 127, 195, 215, 606, 2.27, 8.92, -4.33, -0.08, -0.08, 0.06],
                [132, 127, 195, 215, 606, 2.72, 8.86, -4.22, -0.01, -0.14, 0.04],
                [133, 128, 195, 215, 606, 2.27, 8.84, -4.58, -0.10, -0.12, -0.02],
                [133, 127, 194, 215, 606, 2.18, 8.91, -4.50, -0.05, -0.05, -0.03],
                [132, 127, 195, 215, 607, 2.24, 8.69, -4.49, -0.10, -0.08, 0.02],
                [132, 127, 195, 215, 607, 1.96, 9.05, -4.58, -0.10, -0.13, 0.04],
                [133, 126, 195, 215, 607, 2.26, 8.70, -4.86, -0.08, -0.17, 0.06],
                [133, 127, 195, 215, 607, 2.06, 8.80, -4.87, -0.05, -0.09, -0.00],
                [133, 126, 195, 215, 606, 2.18, 8.82, -4.83, -0.12, -0.05, 0.03],
                [132, 126, 194, 215, 607, 2.24, 8.83, -4.84, -0.05, -0.02, -0.01],
                [133, 126, 195, 215, 607, 2.22, 8.89, -4.87, -0.08, -0.08, 0.00],
                [133, 127, 195, 215, 606, 2.20, 8.88, -4.92, -0.07, -0.09, 0.03],
                [133, 126, 195, 215, 606, 2.22, 8.66, -4.91, -0.04, -0.11, 0.03],
                [133, 126, 195, 215, 607, 2.15, 8.59, -5.30, -0.06, -0.05, 0.01],
                [133, 127, 195, 215, 606, 2.09, 8.71, -5.07, -0.12, -0.08, 0.03],
                [133, 127, 195, 215, 607, 1.98, 8.71, -5.16, -0.04, -0.06, 0.03],
                [133, 126, 195, 215, 607, 2.06, 8.48, -5.33, -0.06, -0.05, 0.04],
                [133, 126, 195, 215, 607, 2.28, 8.69, -4.95, -0.05, -0.04, 0.05],
                [132, 126, 194, 215, 607, 2.22, 8.47, -4.95, -0.08, -0.10, 0.07],
                [133, 126, 195, 215, 606, 2.00, 8.71, -5.18, 0.01, -0.14, 0.10],
                [133, 127, 195, 215, 607, 2.21, 8.64, -5.54, -0.05, -0.10, 0.08],
                [133, 127, 194, 216, 606, 1.86, 8.55, -5.72, -0.10, -0.07, -0.04],
                [133, 127, 195, 215, 607, 1.77, 8.38, -5.68, -0.29, -0.04, 0.00],
                [133, 126, 194, 215, 607, 1.77, 8.58, -5.31, -0.29, -0.07, 0.03],
                [134, 127, 195, 215, 606, 1.97, 8.83, -5.39, -0.21, -0.06, -0.01],
                [133, 126, 195, 215, 607, 2.01, 8.69, -5.25, -0.32, -0.06, -0.05],
                [134, 126, 195, 215, 607, 1.91, 9.03, -5.28, -0.34, 0.00, -0.05],
                [134, 127, 195, 215, 607, 1.89, 8.85, -4.37, -0.18, 0.06, -0.06],
                [134, 126, 194, 215, 606, 1.98, 9.10, -4.52, -0.18, -0.05, 0.04],
                [134, 126, 195, 215, 606, 1.84, 9.03, -4.47, -0.03, -0.00, -0.04],
                [134, 127, 196, 215, 607, 1.98, 8.95, -4.60, -0.26, 0.05, -0.00],
                [134, 126, 194, 215, 607, 1.96, 9.07, -4.51, -0.25, -0.04, -0.01]])

print(b.shape)

b2 = np.array(['B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',])
print(b2.shape)

x_train,y_train = (np.concatenate((a,b)), np.concatenate((a1,b2)))
x_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/predict.txt")
y_test = np.array(['B','A','B','A','B','A'])
predict = np.array([[138, 127, 197, 218, 607, 1.64, 8.84, -4.90, -0.11, -0.01, 0.02],
#[182, 97, 149, 173, 611, 1.71, 9.42, -2.98, - 0.11, - 0.02, 0.01],
[138, 127, 196, 218, 607, 1.53, 8.83, -4.91, -0.10, -0.01, 0.01]

                    ])


print(x_train.shape)
print(y_train.shape)


labelEncoder = LabelEncoder() #instantiating the Label Encoder
y_train_encoded = labelEncoder.fit_transform(y_train)
y_test_encoded = labelEncoder.fit_transform(y_test)
print(y_train_encoded.dtype)


inputs = keras.Input(shape=(11,))
dense_1 = keras.layers.Dense(11, activation='linear')(inputs)
#reshape_1 = keras.layers.Reshape((1, 11))(inputs)
#dense_1 = keras.layers.GRU(11, activation='tanh')
dense_2 = keras.layers.Dense(22,activation='sigmoid')(dense_1)
outputs = keras.layers.Dense(1, activation='linear')(dense_2)
model = keras.Model(inputs=dense_1,outputs=outputs,name='SLTS')

model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['Precision']
)

model.fit(x_train, y_train_encoded, epochs=30)
#model.save('SLTS_model.h5')
predicted_labels = labelEncoder.inverse_transform(model.predict(predict).astype(int).flatten())
#predicted_labels = model.predict(X_test)
print(predicted_labels)