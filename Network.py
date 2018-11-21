import numpy as np 
import tqdm

class Input(): 
	def __init__(self, nodes): 
		self.type = "input"
		self.nodes = nodes
		self.name = "Input Layer"
		self.num_parameters = 0

	def pretty_print(self, verbose=1): 
		s =  self.name + '\t' + '\t' +  str(self.nodes) + '\t' + str(self.num_parameters) 
		if(verbose):
			print(s)
		return s

class Layer(): 
	def __init__(self, nodes):
		self.type = "dense"
		self.nodes = nodes
		self.biases = np.array(np.random.randn(nodes, 1))
		self.weights = None
		self.name = "Fully Connected Layer"
		self.num_parameters = len(self.biases)
		self.activation = Linear()

	def forward(self, inputs, activation=True):
		'''
		if(self.weights): 
			print("Weights not initialized")
			return
		'''
		return self.activation.activate(np.dot(self.weights, inputs) + self.biases) if activation else np.dot(self.weights, inputs) + self.biases

	def activate(self, inputs): 
		return self.activation.activate(inputs)

	def set_weight_dims(self, prev_layer): 
		self.weights = np.array(np.random.randn(self.nodes, prev_layer.nodes))
		self.num_parameters = self.weights.size + self.biases.size

	def pretty_print(self, verbose=1):
		s = self.name + '\t' +  str(self.nodes) + '\t' + str(self.num_parameters) + "\n" + self.activation.pretty_print()
		if(verbose):
			print(s)
		return s

class Activation():
	def __init__(self): 
		self.name = "Not Defined" 

	def activate(self, x): 
		return x
	def activate_prime(self, x): 
		return np.ones_like(x)

	def pretty_print(self, verbose=1): 
		s = self.name + '\t0\t0'
		if(verbose):
			print(s)
		return s

class Linear(Activation): 
	def __init__(self): 
		super()
		self.type = "activation"
		self.name = "Linear Activation"




class Network(): 
	def __init__(self): 
		self.layers = []

	def add_cost_function(self, cost):
		self.cost = cost

	def add(self, layer): 
		if(len(self.layers) == 0): 
			self.layers.append(layer)
		else: 
			layer.set_weight_dims(self.layers[-1])
			self.layers.append(layer)


	def feed_forward(self, inputs_list):
		outputs = list()
		for inputs in inputs_list: 
			output = inputs
			for layer in self.layers[1:]: 
				output = layer.forward(output)
			outputs.append(output)
		return np.array(outputs)

	def stochastic_gradient_descent(self, training_data, epochs=10, batch_size=32, lr=1e-4, test_data=None):
			if(not self.cost): 
				print("Error: Network not compiled with a cost function!")
				return 

			generator = Batch_Generator(training_data, batch_size)
			generator.randomize()
			(x_train_batch, y_train_batch), done = generator.query() 
			while(not done):
				(grad_weights, grad_biases) = self.backprop(x_train_batch, y_train_batch)
				self.weights = self.weights - lr*grad_weights
				self.biases = self.biases - lr*grad_biases
				train_batch, done = generator.query() 


	def backprop(self, x_train_batch, y_train_batch): 
		
		#need to compute and return gradients
		grad_biases = np.array([np.zeros(b.shape) for b in self.biases])
		grad_weights = np.array([np.zeros(w.shape) for w in self.weights])
		activations = list()
		activation_inputs = list()
		activation = x_train_batch
		activations.append(activation)

		for layer in self.layers: 
			activation_input = [layer.feed_forward(x, activation=False) for x in x_train_batch]
			activation_inputs.append(activation_input)
			activation = [layer.activate(weighted_sum) for weighted_sum in activation_input]
			activations.append(activation)

		y_hat_arr = activations[-1] #y_hat_arr = self.feed_forward(x_train_batch)
		cost_prime = np.mean(np.array([self.cost.prime(y_hat_arr[i], y_train_batch[i]) for i in range(len(y_hat_arr))]))
		delta = cost_prime*self.layers[-1].activation.activate_prime(activation_inputs[-1])

		for layer_index in range(1 - len(self.layers), -1)[::-1]:
			activation_input = activation_inputs[layer_index]
			activation_prime = self.layers[layer_index].activation.activate_prime(activation_input)
			delta = np.dot(self.layers[layer_index + 1].weights.transpose(), delta)*activation_prime
			grad_biases[layer_index] = delta
			grad_weights[layer_index] = np.dot(delta, activations[layer_index - 1].transpose())
		return (grad_weights, grad_biases)


class Cost():
	def __init__(self): 
		self.name = "Not Defined"

	def evaluate(self, y_hat, y): 
		print('Warning: Undefined cost function!')
		return 0

	def prime(self, y_hat, y):          
        print('Warning: Undefined cost function!')  
		return 0

class MSE(Cost): 
	def __init__(self): 
		super()
		self.name = "Mean Squared Error"

	def evaluate(self, y_hat, y): 
		return np.mean((y_hat - y)**2)

	def prime(self, y_hat, y): 
		return np.mean(2*(y_hat - y))


class Batch_Generator():
	def __init__(self, data_set, batch_size=32, randomize=False): 
		self.x, self.y = data_set
		self.batch_size = batch_size
		self.current_index = 0
		if(randomize): 
			self.randomize()

	def randomize(self): 
		shuffled_indices = np.random.shuffle(np.array(range(len(self.x))))
		self.x = self.x[shuffled_indices]
		self.y = self.y[shuffled_indices]


	def query(self):
		new_index = self.current_index + self.batch_size
		batch = (self.x[self.current_index: new_index], self.y[self.current_index: new_index])
		finished = False
		if new_index < len(self.x):
			self.current_index = new_index
		else:
			self.current_index = 0
			finished = True
		return batch, finished




'''
net = Network()
net.add(Input(32))
net.add(Layer(64))
'''
