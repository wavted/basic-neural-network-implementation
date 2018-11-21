from Network import Network, Layer, Input, Linear
import numpy as np

model = Network()
model.add(Input(2))
model.add(Layer(3))

inputs = np.empty((2, 1))

#print(model.feed_forward(inputs))

print(model.feed_forward(np.zeros((2, 1))))
print(model.layers[-1].biases)

for layer in model.layers: 
	layer.pretty_print()
