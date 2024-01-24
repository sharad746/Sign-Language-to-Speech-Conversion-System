import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("C:/Users/rikes/OneDrive/Documents/AB_Train.csv")
test_dataset = pd.read_csv("C:/Users/rikes/OneDrive/Documents/AB_Test2.csv")
test = pd.read_csv("C:/Users/rikes/OneDrive/Documents/test.csv")

# create training and testing data
X = dataset.iloc[ : , 0:11]
y = dataset.iloc[ : , 11]
test_X = test_dataset.iloc[ : , 0:11]
test_y = test_dataset.iloc[ : , 11]
X_train, y_train, X_test, y_test = (X, y, test_X, test_y)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

lblEncoder = LabelEncoder()
y_train_encoded = lblEncoder.fit_transform(y_train)
y_test_encoded = lblEncoder.fit_transform(y_test)

# Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=37, p=2, metric='euclidean')
classifier.fit(X_train, y_train_encoded)

testar = np.array([[156,	88,	128, 148, 600, -0.58, 9.57,	-1.86, -0.09, -0.03, -0.02],
                   [181,	97,	149,	172,	610,	1.74,	9.39,	-2.85,	-0.11,	0,	0.01],
[138,	126,	196,	218,	607,	1.62,	8.91,	-4.88,	-0.11,	-0.02,	0.01],
[138,	126,	197,	218,	607,	1.63,	8.89,	-4.87,	-0.11,	-0.01,	0.04]

                   ])
re_testar = testar.reshape(1, -1)
print(re_testar.shape)
# predict the test set results
y_pred =lblEncoder.inverse_transform(classifier.predict(testar))
print(y_pred)




