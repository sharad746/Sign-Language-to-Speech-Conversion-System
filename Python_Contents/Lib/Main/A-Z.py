import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pyttsx3
import serial.tools.list_ports
import numpy as np

#reading data from csv files
dataset = pd.read_csv("D:\TBC\Project\Project\Sign_Language_To_Speech\Datasets/A-Z.csv")

# create training and testing data
X = dataset.iloc[ : , 0:11] #data from all the rows and from column at index 0 to 10 (train dataset)
y = dataset.iloc[ : , 11] #data from all the rows and from column at index 11 (train dataset)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# assigning a numerical value to the target alphabets
labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.fit_transform(y_test)

'''
Define the model: Init K-NN
n_neighbors defines the number of neighbors to take into consideration
p spicifies the power matrix for the Minkwoski distance metric 
p = 2 stands for Euclidean Distance metric and p = 1 is the Manhattan distance metric
'''
model = KNeighborsClassifier(n_neighbors=161, p=2, metric='euclidean')
model.fit(X_train, y_train) #providing the training data

def arduinoConnect():
    serialInst = serial.Serial()    #creating an empty instance of the Serial
    serialInst.baudrate = 9600      #defining the baud rate
    serialInst.port = "COM3"        #connecting to the port COM3
    serialInst.open()               #opening the connection
    serialInst.flushInput()
    num_points = 11
    data = np.zeros(num_points)

    print("Connectimg Arduino.")

    while True:                     #creating an infinite loop
        if serialInst.in_waiting:   #if there is data incoming then
            # read the incoming data
            print(serialInst.readline())
            try:
                data = [float(val) for val in serialInst.readline().decode('utf-8').rstrip().split(' ')]
                print(data)
                data = np.array(data)
            except Exception:
                print(Exception)
            predict(data)

def predict(data):
    # predict the test set results
    #inverse_transform is used since the target variable were previously encoded using LabelEncoder
    y_pred = labelEncoder.inverse_transform(model.predict(data.reshape(1,-1)))
    print("Result = " + y_pred)
    text_to_speech(y_pred)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

arduinoConnect()


