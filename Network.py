import numpy as np
import tqdm
import math
import pickle



#parent activation function class.
#Every activation function has an activate function to compute the activation. 
#an activate_prime function to compute the derivative of the activation, 
#and a pretty_print function that displays information about the activation
class Activation():
    def __init__(self):
        self.name = "Not Defined"

    def activate(self, x):
        return x

    def activate_prime(self, x):
        return np.ones_like(x)

    def pretty_print(self, verbose=1):
        s = self.name + '\t0\t0'
        if (verbose):
            print(s)
        return s

#Below we define the Linear, sigmoid and softmax activations
class Linear(Activation):
    def __init__(self):
        super().__init__()
        self.type = "activation"
        self.name = "Linear Activation"

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.type = "activation"
        self.name = "Sigmoid Activation"

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def activate_prime(self, x):
        f = self.activate(x)
        return f*(1 - f)


class Softmax(Activation): 
    def __init__(self): 
        super()
        self.name = "Softmax"

    def activate(self, x):
        a = np.exp(x)
        return a/np.sum(a)

    def activate_prime(self, y_hat, y): 
        return y_hat - y

#This class defines the input layer.
#Its purpose is to inform the Network class about the dimension of the Input vectors
class Input():
    def __init__(self, nodes):
        self.type = "input"
        #self.nodes is the dimension of the input vector
        self.nodes = nodes
        self.name = "Input Layer"
        self.num_parameters = 0

    def pretty_print(self, verbose=1):
        s = self.name + '\t' + '\t' + str(self.nodes) + '\t' + str(self.num_parameters)
        if (verbose):
            print(s)
        return s

#This class defines a fully connected layer for the neural network
#We can create a layer with arbitrary number of nodes/neurons
#The weight contains its own weights and biases
class Layer():
    def __init__(self, nodes, activation=Linear()):
        self.type = "dense"
        #these are the number of neurons in the layer
        self.nodes = nodes
        #The layer's biases
        self.biases = np.array(np.random.randn(nodes, 1))
        #The layer's weights
        self.weights = None
        self.name = "Fully Connected Layer"
        self.num_parameters = len(self.biases)
        #The layer's activation function
        self.activation = activation

    #This performs forward propagation through the layer. It uses matrix multiplication to obtain the weighted sum
    # z and then optionally passes that through its activation function.
    def forward(self, inputs, activation=True):
        return self.activation.activate(np.dot(self.weights, inputs) + self.biases) if activation else np.dot(
            self.weights, inputs) + self.biases

    def activate(self, inputs):
        return self.activation.activate(inputs)

    def set_weight_dims(self, prev_layer):
        self.weights = np.array(np.random.randn(self.nodes, prev_layer.nodes))
        self.num_parameters = self.weights.size + self.biases.size

    #Displays information about the layer
    def pretty_print(self, verbose=1):
        s = self.name + '\t' + str(self.nodes) + '\t' + str(self.num_parameters) + "\n" + self.activation.pretty_print(
            verbose=0)
        if (verbose):
            print(s)
        return s


class Network():
    def __init__(self):
        self.layers = []

    #Adds a cost function to the layer
    def add_cost_function(self, cost):
        self.cost = cost

    #Adds input and other layers to the network
    def add(self, layer):
        if (len(self.layers) == 0):
            self.layers.append(layer)
        else:
            layer.set_weight_dims(self.layers[-1])
            self.layers.append(layer)

    #This implements the forward propagation. It is implemented by calling the forward function 
    # of each of the layers in the network
    def feed_forward(self, inputs_list):
        outputs = list()
        #we reshape the dimension of the inputs for compatibility with our layers' forward functions
        inputs_list = inputs_list.reshape((inputs_list.shape[0], inputs_list.shape[1], 1))
        #for each layer we forward propagate
        for inputs in inputs_list:
            output = inputs
            for layer in self.layers[1:]:
                output = layer.forward(output)
            outputs.append(output)
        #we reshape the outputs again, to make them more user friendly
        outputs = np.array([o.reshape((len(o),)) for o in outputs])
        return outputs

    #This function displays information about the network
    def summary(self):
        s = "------------------------------------------\n"
        s += "Layer Name\t\tUnits\tParameters\n"
        s += "------------------------------------------\n"
        total_params = 0
        for layer in self.layers: 
            s += layer.pretty_print(verbose=0) + '\n'
            total_params += layer.num_parameters
        s += "------------------------------------------\n"
        s += "Total params: " +  str(total_params) + '\n'
        s += "------------------------------------------\n"
        return s

    #This implements stochastic gradient descent to train the neural network
    def stochastic_gradient_descent(self, training_data, epochs=50, batch_size=32, lr=1e-7, test_data=None, freq=1, cat_eval=False):
        if (not self.cost):
            print("Error: Network not compiled with a cost function!")
            return
        #We train for the specified number of epochs in this for loop
        for i in range(epochs):
            #We instantiate the Batch_Generator to obtain the dataset in batches of size batch_size
            generator = Batch_Generator(training_data, batch_size)
            #randomize the order of the dataset
            generator.randomize()
            #we query the generator to obtain a batch of the data
            (x_train_batch, y_train_batch), done = generator.query()

            #This loop ends when we have finished iterating over the whole dataset once
            while (not done):
                #We use the backprop function to obtain the gradients with respect to all the weights and biases
                (grad_weights, grad_biases) = self.backprop(x_train_batch, y_train_batch, cat_eval=cat_eval)
                
                #We perform the stochastic gradient descent update rules for each layer
                for j in range(len(self.layers))[1:]:
                    #update rules according to equations 6 in the report
                    self.layers[j].weights = self.layers[j].weights - lr * grad_weights[j - 1]
                    self.layers[j].biases = self.layers[j].biases - lr * grad_biases[j - 1]
                
                #Query the next batch of data 
                (x_train_batch, y_train_batch), done = generator.query()

            #This evaluates the model on the test_data if test_data every freq number of epochs
            #This way we can get a feel for how the model is training as we are training it.
            if(test_data and i%freq == 0):
                loss = 0 
                y_hats = self.feed_forward(test_data[0])
                for k in range(len(test_data[0])):
                    loss += self.cost.evaluate(y_hats[k], test_data[1][k])
                loss /= len(test_data[0])
                print("Epoch: " + str(i + 1), "Loss: " + str(loss))
                if(cat_eval): 
                    print("Acc: ", self.categorical_evaluate(test_data))

    #This is a helper function that evaluates the accuracy for multiclass outputs such as in classification
    def categorical_evaluate(self, test_data):
        y_hats =self.feed_forward(test_data[0])
        y_hats = np.array([np.argmax(y_hat) for y_hat in y_hats])
        s = 0
        for i in range(len(y_hats)): 
            if(y_hats[i] == np.argmax(test_data[1][i])): 
                s += 1
        return s/len(y_hats)

    def backprop(self, x_train_batch, y_train_batch, cat_eval=False):
        batch_size = len(x_train_batch)
        grad_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        grad_weights = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        for x, y in zip(x_train_batch, y_train_batch):
            grad_weights_new, grad_biases_new = self.backprop_one(x, y, cat_eval=cat_eval)
            grad_weights = [grad_weights[i] + grad_weights_new[i] for i in range(len(grad_weights))]
            grad_biases = [grad_biases[i] + grad_biases_new[i] for i in range(len(grad_biases))]
        grad_weights = [layer_grad_weights / batch_size for layer_grad_weights in grad_weights]
        grad_biases = [layer_grad_biases / batch_size for layer_grad_biases in grad_biases]

        return (grad_weights, grad_biases)

    def backprop_one(self, x, y, cat_eval=False):
        x = x.reshape((len(x), 1))
        grad_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        grad_weights = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        activations = list()
        activation_inputs = list()
        activation = x
        activations.append(activation)

        for layer in self.layers[1:]:
            activation_input = layer.forward(activations[-1], activation=False)
            activation_inputs.append(activation_input)
            activation = layer.activate(activation_inputs[-1])
            activations.append(activation)


        y_hat = activations[-1]
        if not cat_eval:
            cost_prime = self.cost.prime(y_hat, y)
            delta = cost_prime * self.layers[-1].activation.activate_prime(activation_inputs[-1])
        else: 
            delta = activations[-1] - y.reshape((len(y), 1))

        grad_biases[-1] = delta
        grad_weights[-1] = np.dot(delta, activations[-2].transpose())

        for layer_index in range(1 - len(self.layers), -1)[::-1]:
            activation_input = activation_inputs[layer_index]
            activation_prime = self.layers[layer_index].activation.activate_prime(activation_input)
            delta = np.dot(self.layers[layer_index + 1].weights.transpose(), delta) * activation_prime
            grad_biases[layer_index] = delta
            grad_weights[layer_index] = np.dot(delta, activations[layer_index - 1].transpose())
        return (grad_weights, grad_biases)

    #This saves the network as a pkl file
    def save(self, name): 
        with open(name, 'wb') as output: 
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


#This is the parent cost function which defines an interface. 
#Every cost function must have an evaluate function to evaluate the cost
# and a prime function that implements its derivative 
class Cost():
    def __init__(self):
        self.name = "Not Defined"

    def evaluate(self, y_hat, y):
        print('Warning: Undefined cost function!')
        return 0

    def prime(self, y_hat, y):
        print('Warning: Undefined cost function!')
        return 0

#Below we have defined Mean squared error and Cross-Entropy Cost functions

class MSE(Cost):
    def __init__(self):
        super()
        self.name = "Mean Squared Error"

    def evaluate(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def prime(self, y_hat, y):
        return np.mean(2 * (y_hat - y))


class Cross_Entropy(Cost): 
    def __init__(self): 
        self.name = "Cross Entropy"

    def evaluate(self, y_hat, y): 
        return np.sum(np.nan_to_num(np.where(y == 1, -np.log(y_hat), -np.log(1 - y_hat))))

    #dummy derivative, since we do not use this in our implementation
    def prime(self, y_hat, y): 
        return 0


#This class performs data processing such as randomization and division into batches of equal size
class Batch_Generator():
    def __init__(self, data_set, batch_size=32, randomize=False):
        self.x, self.y = data_set
        self.batch_size = batch_size
        self.current_index = 0
        if (randomize):
            self.randomize()

    #This randomizes the order of the dataset
    def randomize(self):
        shuffled_indices = np.array(range(len(self.x)))
        np.random.shuffle(shuffled_indices)
        self.x = self.x[shuffled_indices]
        self.y = self.y[shuffled_indices]

    #This returns a batch of the dataset of size self.batch_size
    #It also returns a boolean value finished which indicates whether we have 
    #finished iterating over the whole dataset
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

