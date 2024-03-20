import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'],'g',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.xlim([0,100])
    plt.ylim([0,300])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data(x_data,y_data,x,y,title):
    plt.figure(figsize=(15,5))
    plt.scatter(x_data,y_data,label='Ground Truth',color='green',alpha=0.5)
    plt.plot(x,y,color='k',label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$k]')
    plt.title(True)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

SEED_VALUE = 42
#Fix seed to make training deteterministic
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
#Load data housing bostos
(x_train,y_train),(x_test,y_test) = boston_housing.load_data(test_split=0.2)
#div data # Dividir los datos en 80% para entrenamiento y 20% para pruebas
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
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

#create model type sequential
model = Sequential()
#define model using single neuron y=mx+b
model.add(Dense(1,input_shape=(1,)))
#show features neuronal networking
model.summary()

#compile model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005),loss='mse')

#view history model
history = model.fit(
    x_train_1d,
    y_train,
    batch_size=16,
    epochs=101,
    validation_split=0.3,
)
#graph loss
plot_loss(history)

#create 10 number data with values of 3 - 9
x = np.linspace(3,9,10)

#predict values and set x
y = model.predict(x)
plot_data(x_train_1d,y_train,x,y,title='Training Dataset')
plot_data(x_test_1d,y_test,x,y,title='Test Dataset')

model.save('model/model.h5')