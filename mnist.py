from Network import Network, Layer, Input, Linear, MSE, Sigmoid
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical 


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))
print(x_test.shape)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Network()
model.add(Input(784))
#model.add(Layer(30, activation=Sigmoid()))
model.add(Layer(10, activation=Sigmoid()))


model.add_cost_function(MSE())


print(model.summary())

inputs = x_test[0:10]


print(model.feed_forward(inputs))
#y_hats = np.array([np.argmax(y_hat) for y_hat in model.feed_forward(inputs)])
#print(y_hats)
print(y_test[0:10])

model.stochastic_gradient_descent((x_train, y_train), epochs=50, lr=3e-4, test_data=(x_test, y_test), cat_eval=True, freq=2)
print(model.feed_forward(inputs))
print(y_test[0:10])


