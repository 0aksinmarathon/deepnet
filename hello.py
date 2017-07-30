import numpy as np

class Network:
    def __init__(self, layers, layer_n):
        self.layers = layers
        self.layer_n = layer_n


    def minibatch_training(self):
        pass


    def feedforward(self, input):
        op = input
        for layer in self.layers:
            layer.calc(op)
            op = layer.output
        print(op)

    def online_bp(self, inputs, training_data, learning_rate, drop_out, regularization):
        for i in range(len(inputs)):
            input = inputs[i]
            correct = training_data[i]
            self.feedforward(input)
            op_layer = self.layers[-1]
            op_layer.delta = op_layer.output - correct
            op_layer.dif = np.matmul(op_layer.delta.reshape(len(op_layer.delta), 1),
                                                  op_layer.input.reshape(1, len(op_layer.input)))
            op_layer.w = op_layer.w - learning_rate * op_layer.dif
            prev_layer = op_layer
            for layer in reversed(self.layers[:-1]):
                layer.delta = sigmoid_grad(layer.output) * np.matmul(prev_layer.w.T, prev_layer.delta)
                layer.dif = np.matmul(layer.delta.reshape(len(layer.delta), 1), layer.input.reshape(1, len(layer.input)))
                layer.w = layer.w - learning_rate * layer.dif
                prev_layer = layer
                #print(layer.w)

                #print("""
                 #   ~~~~~~~~~~~~~
                  ## ~~~~~~~~~~~~~~
                    #""" % (i + 1))


class ConnectedLayer:
    def __init__(self, input_n, output_n, activation):
        self.input_n = input_n
        self.output_n = output_n
        self.w = np.random.rand(self.output_n, self.input_n)
        self.b = np.random.rand()
        self.activation = activation
    def calc(self, input):
        self.input = input
        self.output = self.activation(np.matmul(self.w, self.input)) + self.b
def sigmoid(x):
    return 1/1+np.e**(-x)

def identity(x):
    return x

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


net = Network([ConnectedLayer(5, 3, sigmoid),
               ConnectedLayer(3, 10, sigmoid),
               ConnectedLayer(10, 2, identity)], 3)
ip = np.ones((100, 5))
td = np.ones((100, 2))
net.online_bp(ip, td, 0.01, False, False)



