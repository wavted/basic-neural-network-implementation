from Network import Network, Layer, Input, Linear, MSE
import numpy as np
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

model = Network()
model.add(Input(13))
model.add(Layer(1))


model.add_cost_function(MSE())

for layer in model.layers: 
	layer.pretty_print()

inputs = x_test[0]

#print(model.feed_forward(inputs))

print(model.feed_forward(inputs)[0])
print(y_test[0])

model.stochastic_gradient_descent((x_train, y_train))
print(model.feed_forward(inputs)[0])
print(y_test[0])


