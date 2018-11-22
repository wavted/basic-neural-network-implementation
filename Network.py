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
        s = self.name + '\t' +  str(self.nodes) + '\t' + str(self.num_parameters) + "\n" + self.activation.pretty_print(verbose=0)
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
        inputs_list = inputs_list.reshape((inputs_list.shape[0], inputs_list.shape[1], 1))
        for inputs in inputs_list: 
            #output = inputs.reshape(len(inputs), 1)
            output = inputs
            for layer in self.layers[1:]: 
                output = layer.forward(output)
                print(output.shape)
            outputs.append(output)
        return np.array(outputs)

    def stochastic_gradient_descent(self, training_data, epochs=200, batch_size=32, lr=1e-4, test_data=None):
            if(not self.cost): 
                print("Error: Network not compiled with a cost function!")
                return 

            generator = Batch_Generator(training_data, batch_size)
            generator.randomize()
            (x_train_batch, y_train_batch), done = generator.query() 
            while(not done):
                (grad_weights, grad_biases) = self.backprop(x_train_batch, y_train_batch)
                #for w in grad_weights:
                #    print(w)
                #for l in self.layers[1:]: 
                #    print(l.weights.shape)
                for i in range(len(self.layers))[1:]:
                    self.layers[i].weights = self.layers[i].weights - lr*grad_weights[i -1]
                    self.layers[i].biases = self.layers[i].biases - lr*grad_biases[i -1]
                train_batch, done = generator.query() 

    '''
    def backprop(self, x_train_batch, y_train_batch): 
        
        #need to compute and return gradients
        grad_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        grad_weights = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        activations = list()
        activation_inputs = list()
        activation = x_train_batch
        activations.append(activation)

        for layer in self.layers[1:]: 
            activation_input = [layer.forward(x, activation=False) for x in activations[-1]]
            activation_inputs.append(activation_input)
            activation = [layer.activate(weighted_sum) for weighted_sum in activation_input[-1]]
            activations.append(activation)

        activation_inputs = activation_inputs
        #print(activations)

        #y_hat_arr = [activations[layer_index][-1] for layer_index in range(len(self.layers))[1:]] #y_hat_arr = self.feed_forward(x_train_batch)
        y_hat_arr = activations[-1]
        cost_prime = np.mean(np.array([self.cost.prime(y_hat_arr[i], y_train_batch[i]) for i in range(len(y_hat_arr))]))
        delta = cost_prime*self.layers[-1].activation.activate_prime(activation_inputs[-1])
        print(delta.shape)
        for layer_index in range(1 - len(self.layers), -1)[::-1]:
            activation_input = activation_inputs[layer_index]
            activation_prime = self.layers[layer_index].activation.activate_prime(activation_input)
            print("here: ", activation_prime.shape)
            delta = np.dot(self.layers[layer_index + 1].weights.transpose(), delta)*activation_prime
            grad_biases[layer_index] = delta
            grad_weights[layer_index] = np.dot(delta, activations[layer_index - 1].transpose())

        print(grad_weights, grad_biases)
        return (grad_weights, grad_biases)
    '''

    def backprop(self, x_train_batch, y_train_batch):
        batch_size = len(x_train_batch)
        grad_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        grad_weights = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        for x, y in zip(x_train_batch, y_train_batch): 
            grad_weights_new, grad_biases_new = self.backprop_one(x, y)
            #grad_weights += grad_weights_new
            #grad_biases += grad_biases_new
            grad_weights = [grad_weights[i] + grad_weights_new[i] for i in range(len(grad_weights))]
            grad_biases = [grad_biases[i] + grad_biases_new[i] for i in range(len(grad_biases))]
        grad_weights = [layer_grad_weights/batch_size for layer_grad_weights in grad_weights]
        grad_biases = [layer_grad_biases/batch_size for layer_grad_biases in grad_biases]

        return (grad_weights, grad_biases)


    def backprop_one(self, x, y): 
        grad_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        grad_weights = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        activations = list()
        activation_inputs = list()
        activation = x
        activations.append(activation)

        for layer in self.layers[1:]: 
            activation_input = layer.forward(activations[-1], activation=False)
            activation_inputs.append(activation_input)
            activation = layer.activate(activation_input[-1])
            activations.append(activation)

        y_hat = activations[-1]
        cost_prime = self.cost.prime(y_hat, y)
        #print(cost_prime)
        delta = cost_prime*self.layers[-1].activation.activate_prime(activation_inputs[-1])

        for layer_index in range(1 - len(self.layers), -1)[::-1]:
            activation_input = activation_inputs[layer_index]
            activation_prime = self.layers[layer_index].activation.activate_prime(activation_input)
            #print("here: ", activation_prime.shape)
            delta = np.dot(self.layers[layer_index + 1].weights.transpose(), delta)*activation_prime
            grad_biases[layer_index] = delta
            grad_weights[layer_index] = np.dot(delta, activations[layer_index - 1].transpose())
        #print(grad_weights)
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
        shuffled_indices = np.array(range(len(self.x)))
        np.random.shuffle(shuffled_indices)
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
