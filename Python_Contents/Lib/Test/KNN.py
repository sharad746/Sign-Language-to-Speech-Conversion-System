import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from math import sqrt
import scipy.stats

#A Train and test dataset
A_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_train.txt")
A_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_train.txt", dtype='str')
A_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_X_test.txt")
A_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/A_data/A_y_test.txt", dtype='str')


#B train and test dataset
B_X_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_train.txt")
B_y_train = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_train.txt", dtype='str')
B_X_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_X_test.txt")
B_y_test = np.loadtxt("D:/FYP/numberImageRecognition/Lib/train/B_data/B_y_test.txt", dtype='str')

(X_train, y_train),(X_test, y_test) = (np.concatenate((A_X_train,B_X_train)), np.concatenate((A_y_train,B_y_train))), (np.concatenate((A_X_test, B_X_test)), np.concatenate((A_y_test,B_y_test)))

modes = scipy.stats.mode(y_train)
print(modes)
knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(X_train, y_train)
#prediction = knn_model.predict((X_test))
#print(prediction)



