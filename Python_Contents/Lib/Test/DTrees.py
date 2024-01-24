from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

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


# Split the data into training and testing sets
(X_train, y_train),(X_test, y_test) = (np.concatenate((A_X_train,B_X_train)), np.concatenate((A_y_train,B_y_train))), (np.concatenate((A_X_test, B_X_test)), np.concatenate((A_y_test,B_y_test)))

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier
clf.fit(X_train, y_train)

# Evaluate the performance of the decision tree classifier
score = clf.score(X_test, y_test)
print("Test score: ", score)

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
predictions = clf.predict(b)
print(predictions)