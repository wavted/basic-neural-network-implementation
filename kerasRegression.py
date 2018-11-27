import keras
import numpy as np 
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
model = Sequential()
model.add(Dense(13, input_dim=13))
model.add(Dense(1))
optimizer = SGD(lr=1e-6)
model.compile(loss='mse', optimizer=optimizer)

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=.2)