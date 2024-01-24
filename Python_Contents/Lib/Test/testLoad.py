import pandas as pd
import numpy as np

def read():
    with open('test2.txt', 'w') as f:
        ar = [1,2,3,4,5,6,7,8,9]
        f.write(','.join(map(str, ar)))
    n()

def n():
    data = np.loadtxt('test2.txt', delimiter=',')
    print(data)
    print(data.dtype)

read()
