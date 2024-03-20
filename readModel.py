import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

model = load_model('model/model.h5')



#using model with data
x = [3,4,5,6,7]

#predict model
y_pred = model.predict(x)

#for model
for idx in range(len(x)):
    print(f"Predict price of a home with {x[idx]} rooms:${int(y_pred[idx]*10)/10}")

#create 10 number data with values of 3 - 9
x = np.linspace(3,9,10)

#predict values and set x
y = model.predict(x)


