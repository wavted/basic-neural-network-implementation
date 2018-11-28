import numpy as np
#import dill



class Layer:
    def __init__(self, nodes_size, next_nodes_size, activation):
        self.nodes_size= nodes_size
        self.activation = activation
        self.activations = np.zeros([nodes_size,1])
        if next_nodes_size != 0:
            self.layer_weights = np.random.normal(0, 0.001, size=(nodes_size, next_nodes_size))
        else:
            self.layer_weights = None

class Network:

    def __init__(self, cost, layers_size, nodes_size, activation):
        self.layers = []
        self.cost = cost
        self.activation=activation
        #layers_size is scalar
        self.layers_size=layers_size
        #nodes_size is vector
        self.nodes_size=nodes_size
        for i in range(layers_size):
            if i != layers_size-1:
                added_layer=Layer(nodes_size[i],nodes_size[i+1],activation)
            else:
                added_layer=Layer(nodes_size[i] , 0 , activation)
            self.layers.append(added_layer)

        
        
    def stochastic_gradient_descent(self, training_data,  epochs=50, batch_size=32, lr=1e-7, test_data=None, freq=1, cat_eval=False, filename=None):
        if (not self.cost):
            print("Error: Network not compiled with a cost function!")
            return
        for i in range(epochs):
            generator = Batch_Generator(training_data, batch_size)
            generator.randomize()
            train_batch, done = generator.query()

            while (not done):
                self.gradient_descent(train_batch, lr, test_data, freq, cat_eval)
            
            if(test_data and i%freq == 0):
                loss = 0 
                y_hats = self.feed_forward(test_data[0])
                for k in range(len(test_data[0])):
                    loss += self.cost.evaluate(y_hats[k], test_data[1][k])
                loss /= len(test_data[0])
                print("Epoch: " + str(i + 1), "Loss: " + str(loss))
                if(cat_eval): 
                    print("Acc: ", self.categorical_evaluate(test_data))
        
        if filename:
            dill.dump_session(filename)
        


    def gradient_descent(self, training_data, lr=1e-7, test_data=None, freq=1, cat_eval=False):
        self.lr = lr
        inputs , targets = training_data
        self.error = 0 
        self.feed_forward(inputs)
        self.calculate_loss(targets)
        self.back_prop(targets)

        

    def feed_forward(self, inputs):
        self.layers[0].activations = inputs
        #print(len(self.layers))
        #print(self.layers[self.layers_size-1].activations)
        print(self.layers[self.layers_size-1].layer_weights)
        temp=np.dot(self.layers[self.layers_size-1].activations, self.layers[self.layers_size-1].layer_weights)
        for i in range(self.layers_size-1):
            if self.layers[i+1].activation == "sigmoid":
                self.layers[i+1].activations = self.sigmoid(temp)
            elif self.layers[i+1].activation == "softmax":
                self.layers[i+1].activations = self.softmax(temp)
            elif self.layers[i+1].activation == "relu":
                self.layers[i+1].activations = self.relu(temp)
            elif self.layers[i+1].activation == "tanh":
                self.layers[i+1].activations = self.tanh(temp)
            else:
                self.layers[i+1].activations = temp

    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(layer)))

    def tanh(self, layer):
        return np.tanh(layer)

    def calculate_loss(self, targets):
        if len(targets[0]) != self.layers[self.layers_size-1].nodes_size_in_layer:
            print ("Error: Label is not of the same shape as output layer.")
            print("Label: ", len(targets), " : ", len(targets[0]))
            print("Out: ", len(self.layers[self.layers_size-1].activations), " : ", len(self.layers[self.layers_size-1].activations[0]))
            return
        
        if self.cost_function == "mean_squared":
            self.error = np.mean(np.divide(np.square(np.subtract(targets, self.layers[self.layers_size-1].activations)), 2))
        elif self.cost_function == "cross_entropy":
            self.error = np.negative(np.sum(np.multiply(targets, np.log(self.layers[self.layers_size-1].activations))))

    def back_prop(self, targets):

        y = self.layers[self.layers_size-1].activations
        deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, targets-y)))
        new_weights = self.layers[i-1].layer_weights - self.learning_rate * deltaw
        for i in range(self.layers_size-2, 0, -1):
            y = self.layers[i].activations
            deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_weights, self.layers[i].layer_weights),axis=1).T)))
            self.layers[i].layer_weights = new_weights
            new_weights = self.layers[i-1].layer_weights - self.learning_rate * deltaw
        self.layers[0].layer_weights = new_weights
    
    def predict(self,  input, filename=None):

        self.batch_size = 1
        self.forward_pass(input)
        a = self.layers[self.layers_size-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        return a

    def check_accuracy(self, inputs, targets, filename=None):
        if filename:
            dill.load_session(filename)
        self.batch_size = len(inputs)
        self.forward_pass(inputs)
        a = self.layers[self.layers_size-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        total=0
        correct=0
        for i in range(len(a)):
            total += 1
            if np.equal(a[i], targets[i]).all():
                correct += 1
        print("Accuracy: ", correct*100/total)



    def load_model(self, filename):
        dill.load_session(filename)
        

class Batch_Generator():
    def __init__(self, data_set, batch_size=32, randomize=False):
        #print(data_set)
        self.x, self.y = data_set
        self.batch_size = batch_size
        self.current_index = 0
        if (randomize):
            self.randomize()

    def randomize(self):
        shuffled_indices = np.array(range(len(self.x)))
        np.random.shuffle(shuffled_indices)
        self.x = self.x[shuffled_indices]
        self.y = self.y[shuffled_indices]
        #print(self.x)
        #print(self.y)

    def query(self):
        new_index = self.current_index + self.batch_size
        batch = (self.x[self.current_index: new_index], self.y[self.current_index: new_index])
        #print(batch)
        finished = False
        if new_index < len(self.x):
            self.current_index = new_index
        else:
            self.current_index = 0
            finished = True
        return batch, finished