from Network import Network, Layer, Input, Linear, MSE
import numpy as np
from keras.datasets import boston_housing

#Load the dataset and normalize it
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train = (x_train - np.mean(x_train))/np.std(x_train)
x_test = (x_test -np.mean(x_test))/np.std(x_test)

#Define the linear regression model by adding the input layer and fully connected layers
#By default the activation for the layers is linear
model = Network()
model.add(Input(13))
model.add(Layer(2))
model.add(Layer(2))
model.add(Layer(1))

#Above we have 3 layers. For the one layer case we just comment out the two layers in the middle.

#Add the mean squared error cost function to the network 
model.add_cost_function(MSE())

#Print a nicely formatted description of the network
print(model.summary())

inputs = x_test[0:10]

#Test the model on selected inputs pre-training
print(model.feed_forward(inputs))
print(y_test[0:10])

#Train the model by calling stochastic gradient descent
model.stochastic_gradient_descent((x_train, y_train), epochs=5000, lr=1e-3, test_data=(x_test, y_test), freq=100)

#Test the model post training
print(model.feed_forward(inputs))
print(y_test[0:10])


