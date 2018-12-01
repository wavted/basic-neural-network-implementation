from Network import Network, Layer, Input, Linear, MSE, Sigmoid, Softmax, Cross_Entropy, ReLU
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical 

#We load the datset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Flatten the input (28, 28) images to 784 dimensional vectors 
#and normalize the data by divinding by max pixel value 255
x_train = x_train.reshape((len(x_train), 784))/255
x_test = x_test.reshape((len(x_test), 784))/255

#Convert the labels into one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Create the neural network using our Network class
model = Network()
#The input images are 28 by 28. Since we reshaped it into a 1D array of size 784, 
#the input layer is 784 dimensional.
model.add(Input(784))
#We have only one fully connected layer since this is logistic regression
model.add(Layer(10, activation=Sigmoid()))

#We add the Cross_Entropy cost function our network
model.add_cost_function(Cross_Entropy())

#Print out description of the network
print(model.summary())

inputs = x_test[0:10]

#Test on some inputs before training
print(model.feed_forward(inputs))

print(y_test[0:10])

#Train the network
model.stochastic_gradient_descent((x_train, y_train), epochs=25, lr=3, test_data=(x_test, y_test), cat_eval=True, freq=1)

#Test on selected inputs after training. 
print(model.feed_forward(inputs))
print(y_test[0:10])

#Save the trained model
model.save('mnist.pkl')

