from Network import Network, Layer, Input, Linear, MSE
import numpy as np
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train = (x_train - np.mean(x_train))/np.std(x_train)
x_test = (x_test -np.mean(x_test))/np.std(x_test)
#print(x_test.shape)

model = Network()
model.add(Input(13))
model.add(Layer(13))
model.add(Layer(1))


model.add_cost_function(MSE())
'''
for layer in model.layers: 
	layer.pretty_print()
'''

print(model.summary())

inputs = x_test[0:10]

#print(model.feed_forward(inputs))

print(model.feed_forward(inputs))
print(y_test[0:10])

#model.stochastic_gradient_descent(((x_train - np.mean(x_train))/np.std(x_train), y_train), epochs=5000, lr=1e-4, test_data=((x_test -np.mean(x_test))/np.std(x_test), y_test), freq=100)
model.stochastic_gradient_descent((x_train, y_train), epochs=5000, lr=1e-3, test_data=(x_test, y_test), freq=100)
print(model.feed_forward(inputs))
print(y_test[0:10])


