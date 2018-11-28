from rewrite_network import Network, Layer, Batch_Generator

import numpy as np
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
#print(x_train)

model = Network("mean_squared", 3, (13,5,1), "relu")


'''
for layer in model.layers: 
	layer.pretty_print()
'''


#inputs = x_test[0:10]

#print(model.feed_forward(inputs))

#print(model.feed_forward(inputs))
#print(y_test[0:10])

model.stochastic_gradient_descent((x_train, y_train), epochs=5000, lr=1e-6, test_data=(x_test, y_test), freq=100)
#print(model.feed_forward(inputs))
#print(y_test[0:10])


