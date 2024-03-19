import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt

SEED_VALUE = 42
#Fix seed to make training deteterministic
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
#Load data housing bostos
(x_train,y_train),(x_test,y_test) = boston_housing.load_data()
#feature selected model # colum is five
boston_feature = {
    'Average Number of Rooms':5,
}
#one feature
x_train_1d = x_train[:,boston_feature['Average Number of Rooms']]
x_test_1d = x_test[:,boston_feature['Average Number of Rooms']]
#graphic relation
plt.figure(figsize=(15,5))
plt.xlabel('Average Number of Rooms')
plt.ylabel('Media price price [$K]')
plt.grid("on")
plt.scatter(x_train_1d,y_train,color = 'green',alpha=0.5)
plt.show()
