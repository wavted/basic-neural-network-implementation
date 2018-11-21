from Network import Network, Layer, Input, Linear, MSE
import numpy as np

model = Network()
model.add(Input(2))
model.add(Layer(3))


model.add_cost_function(MSE())

inputs = np.array([np.empty((2, 1))])

#print(model.feed_forward(inputs))

print(model.feed_forward(inputs))
print(model.layers[-1].biases)

model.stochastic_gradient_descent((inputs, np.array([[1, 0, 0]])))
print(model.feed_forward(inputs))

for layer in model.layers: 
	layer.pretty_print()
