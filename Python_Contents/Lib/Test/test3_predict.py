import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder

a_data = np.array([[180, 98, 146, 171, 609, 0.57, 2.44, -0.46, -0.02, -0.05, 0.00],
            [180, 97, 149, 172, 610, 2.09, 9.21, -3.59, 0.04, -0.20, 0.05],
            [182, 96, 148 ,172, 609, 1.85, 9.29, -3.05, -0.16, -0.10, 0.05],
            [180, 96, 147 ,171, 610, 1.71, 9.29, -3.23, -0.11, -0.00, -0.01],
            [181, 96, 147 ,172, 610, 1.88, 9.29, -2.70, -0.09, 0.05, -0.01],
            [180, 97, 147 ,172, 609, 1.79, 9.30, -2.77 -0.20 ,-0.02, 0.11],
            [181, 97, 148 ,171, 610, 1.82, 9.36, -3.29, -0.04, -0.05, -0.03],
            [181, 97, 148 ,172, 610, 1.64, 9.36, -2.83, -0.05, -0.02, -0.05],
            [182, 98 ,148 ,172, 610, 1.94, 9.31, -3.28, -0.10, 0.05, 0.03],
            [180, 96, 147 ,170, 609, 1.70, 9.26, -3.11, -0.03, 0.08, 0.01]])

print(a_data.shape)
a2 = np.array(['a','a','a','a','a','a','a','a','a','a'])

b = np.array([[132, 127, 195, 214, 606, 0.58, 2.24, -1.08, -0.03, -0.00, 0.00],
                [132, 127, 194, 215, 607, 2.26, 8.97, -4.28, -0.13, -0.01, 0.01],
                [132, 127, 194, 215, 607, 2.32, 8.99, -4.24, -0.10, 0.00, 0.01],
                [133, 127, 194, 214, 606, 2.24, 8.95, -4.31, -0.10, -0.01, 0.01],
                [132, 127, 194, 214, 606, 2.34, 8.98, -4.12, -0.10, -0.01, 0.02],
                [132, 126, 194, 214, 607, 2.30, 8.97, -4.19, -0.12, -0.02, 0.02],
                [132, 127, 194, 215, 606, 2.26, 9.05, -4.11, -0.11, -0.01, 0.03],
                [132, 126, 194, 214, 606, 2.29, 9.03, -4.17, -0.11, -0.01,0.01],
                [132, 127, 195, 215, 607, 2.41, 8.96, -4.30, -0.10, -0.02, 0.02],
                [132, 126, 195, 215, 607, 2.36, 8.98, -4.23, -0.09, -0.02, 0.02]])
print(b.shape)
b2 = np.array(['b','b','b','b','b','b','b','b','b','b'])

(X_train, y_train) =(np.concatenate((a_data,b)), np.concatenate((a2,b2)))

labelEncoder = LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(y_train)

inputs = keras.Input(shape=(11,))
reshape1 = keras.layers.Reshape((1,11))(inputs)
dense_1 = keras.layers.GRU(45, activation = 'sigmoid')(reshape1)
outputs = keras.layers.Dense(1, activation='linear')(dense_1)
model = keras.Model(inputs=inputs,outputs=outputs)

model.compile (optimizer='Adadelta',
               loss='mse',
               metrics='accuracy'

               )

model.fit(X_train,y_train,epochs=50)
predicted = model.predict(np.array([136, 126, 196, 216, 607, 1.65, 8.85, -4.61, -0.12, -0.02, 0.03]))
print(predicted)