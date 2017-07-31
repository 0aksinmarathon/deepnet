import numpy as np

class Network:
    def __init__(self, layers, layer_n):
        self.layers = layers
        self.layer_n = layer_n


    def minibatch_training(self):
        pass


    def feedforward(self, input, do_rate = 1):
        op = input
        for layer in self.layers:
            #layer.dropout = np.random.binomial(p=do_rate, n=1, size=layer.w.shape)
            #layer.dropped_out = (np.ones(layer.w.shape) - layer.dropout) * layer.w
            #layer.w = layer.dropout * layer.w
            layer.calc(op)
            op = layer.output
        print(op)

    def online_bp(self, inputs, t_data, l_rate1, method, do_rate = 1, reg_rate = 0.0, weight_limit = 0.0, l_rate2 = 0.0):
        if method == self.GD:
            method(inputs, t_data, l_rate1, do_rate, reg_rate)
        if method == self.momentum:
            method(inputs, t_data, l_rate1, l_rate2)
        if method == self.AdaGrad:
            method(inputs, t_data, l_rate1)


            #############################

    def GD(self, inputs, t_data, l_rate, do_rate, reg_rate):
        print("GD method")
        for i in range(len(inputs)):
            input = inputs[i]
            correct = t_data[i]
            self.feedforward(input, do_rate)
            for layer in reversed(self.layers):
                layer.delta = layer.output - correct if layer == self.layers[-1] \
                         else layer.activation(layer.output, prime=True) * np.matmul(prev_layer.w.T, prev_layer.delta)
                layer.dif = l_rate * np.matmul(layer.delta.reshape(len(layer.delta), 1), layer.input.reshape(1, len(layer.input))) + \
                            reg_rate * np.linalg.norm(layer.w)
                layer.w = layer.w - layer.dif
                prev_layer = layer
                #print(layer.w)
            for layer in self.layers:
                layer.w = layer.w * layer.dropout + layer.dropped_out

    def AdaGrad(self, inputs, t_data, l_rate):
        print("あだぐら")
        for i in range(len(inputs)):

            input = inputs[i]
            correct = t_data[i]
            self.feedforward(input)

            op_layer = self.layers[-1]
            op_layer.delta = op_layer.output - correct

            op_layer.dif = np.matmul(op_layer.delta.reshape(len(op_layer.delta), 1),
                                     op_layer.input.reshape(1, len(op_layer.input)))
            op_layer.w = op_layer.w - l_rate * op_layer.dif
            prev_layer = op_layer
            #while np.linalg.norm(op_layer.w) > 100:
            #    op_layer.w = 0.9 * op_layer.w

            for layer in reversed(self.layers[:-1]):
                layer.delta = layer.activation(layer.output, prime=True) * np.matmul(prev_layer.w.T, prev_layer.delta)
                layer.dif = np.matmul(layer.delta.reshape(len(layer.delta), 1), layer.input.reshape(1, len(layer.input)))
                for j in range(len(layer.w)):
                    layer.w[j] = layer.w[j] - (l_rate * layer.dif[j] / np.linalg.norm(layer.w[j]))
                layer.w = layer.w - l_rate * layer.dif
                prev_layer = layer

                #while np.linalg.norm(layer.w) > 100:
                #    layer.w = 0.9 * layer.w







    def momentum(self, inputs, t_data, l_rate1, l_rate2):
        print("momentum method")
        for i in range(len(inputs)):
            input = inputs[i]
            correct = t_data[i]
            self.feedforward(input)
            op_layer = self.layers[-1]
            op_layer.prev_dif = op_layer.dif if i != 0 else 0


            op_layer.delta = op_layer.output - correct
            op_layer.dif =  np.matmul(op_layer.delta.reshape(len(op_layer.delta), 1),
                                     op_layer.input.reshape(1, len(op_layer.input)))
            op_layer.w = op_layer.w - l_rate1 * op_layer.dif + l_rate2 * op_layer.prev_dif
            prev_layer = op_layer

            for layer in reversed(self.layers[:-1]):
                layer.delta = layer.activation(layer.output, prime=True) * np.matmul(prev_layer.w.T, prev_layer.delta)
                layer.prev_dif = layer.dif if i != 0 else 0
                layer.dif = np.matmul(layer.delta.reshape(len(layer.delta), 1), layer.input.reshape(1, len(layer.input)))
                layer.w = layer.w - (l_rate1 * layer.dif) + (l_rate2 * layer.prev_dif)
                prev_layer = layer



class FullyConnectedLayer:
    def __init__(self, input_n, output_n, activation):
        self.input_n = input_n
        self.output_n = output_n
        self.w = np.random.rand(self.output_n, self.input_n)
        self.b = np.random.rand()
        self.activation = activation
    def calc(self, input):
        self.input = input

        self.output = self.activation(np.matmul(self.w, self.input) + self.b)


class ConvolutionalLayer2D:
    def __init__(self, input_w, input_h,  activation,  filter_w, filter_h, stride = 1, pad = 0, pad_method = 'mean'):
        self.input_w = input_w
        self.input_h = input_h
        self.input_n = input_w * input_h
        self.output_w = ((input_w - filter_w + 2 * pad) / stride) + 1
        self.output_h = ((input_h - filter_h + 2 * pad) / stride) + 1
        self.output_n = (((input_h - filter_h) / stride) + 1) * (((input_w - filter_w) / stride) + 1)
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.stride = stride
        self.pad = pad
        self.pad_method = pad_method
        self.w = np.random.rand(filter_h, filter_w)
        self.activation = activation

    def calc(self, input):
        self.input = np.pad(input.reshape(self.input_h, self.input_w), self.pad, self.pad_method)
        print(input)
        self.output = np.array([[(self.w * self.input[j * self.stride:j * self.stride + self.filter_h, \
                                             i * self.stride:i * self.stride + self.filter_w]).sum() \
                        for i in range(int(self.output_w))] for j in range(int(self.output_h))])
        print(self.output)
class ConvolutionalLayer3D:
    def __init__(self, input_w, input_h, input_d, activation,  filter_w, filter_h, filter_d, stride = 1, pad = 0):
        self.input_w = input_w
        self.input_h = input_h
        self.input_d = input_d
        self.input_n = input_w * input_h * input_d
        self.output_w = ((input_w - filter_w + 2 * pad) / stride) + 1
        self.output_h = ((input_h - filter_h + 2 * pad) / stride) + 1
        self.output_d = ((input_d - filter_d + 2 * pad) / stride) + 1
        self.output_n = (((input_h - filter_h) / stride) + 1) * (((input_w - filter_w) / stride) + 1) * (((input_d - filter_d) / stride) + 1)
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.filter_d = filter_d
        self.stride = stride
        self.w = np.random.rand(filter_h, filter_w)

        self.activation = activation

    def calc(self, input):
        self.input = np.pad(input.reshape(self.input_h, self.input_w), self.pad, self.pad_method)
        self.output = [[[(self.w * self.input[i * self.stride: i * self.stride + self.filter_h, \
                                              j * self.stride: j * self.stride + self.filter_w, \
                                              k * self.stride: k * self.stride + self.filter_d]).sum() \
                        for i in range(int(self.output_w))] for j in range(int(self.output_h))] for k in range(int(self.output_d))]

class PoolingLayer:
    def __init__(self, input_w, input_h, filter_w, filter_h, method = 'max'):
        self.input_w = input_w
        self.input_h = input_h
        self.input_n = input_h * input_w
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.pool_method = method
    def calc(self, input):
        self.input = input
        if self.pool_method == 'max':
            self.max(input)
        elif self.pool_method == 'mean':
            self.mean(input)
        elif self.pool_method == 'min':
            self.mean(input)



    def max(self, input):
        self.output = np.array([[np.max(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                        self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])

    def mean(self, input):
        self.output = np.array([[np.mean(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                        self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])


    def min(self, input):
        self.output = np.array([[np.min(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                         self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])

class RectifiedLinearUnit:
    def __init__(self, input_w, input_h):
        self.input_w = input_w
        self.input_h = input_h

    def calc(self, input):
        self.input = input
        negative_to_zero = (input >= 0)
        self.output = self.input * negative_to_zero



def sigmoid(x, prime = False):
    sigma = 1.0 / (1.0 + np.exp(-x))
    if not prime:
        return sigma
    if prime:
        return sigma * (1 - sigma)

def identity(x, prime = False):
    if not prime:
        return x
    if prime:
        return 1

def hinge(x, prime = False):
    if not prime:
        return max(0, x)
    if prime:
        if x > 0:
            return 1
        else:
            return 0






net = Network([FullyConnectedLayer(5, 3, sigmoid),
               FullyConnectedLayer(3, 2, sigmoid),
FullyConnectedLayer(2, 784, sigmoid),
FullyConnectedLayer(784, 28, sigmoid),
FullyConnectedLayer(28, 20, sigmoid),
FullyConnectedLayer(20, 2, sigmoid),
               FullyConnectedLayer(2, 2, identity)], 1)
net2 = Network([ConvolutionalLayer2D(28, 28, identity, 2, 2, stride = 1),
               PoolingLayer(27, 27, 3, 3, 'mean'),
                ConvolutionalLayer2D(9, 9, identity, 4, 4, stride=1),
                PoolingLayer(6, 6, 3, 3, 'mean'),
                RectifiedLinearUnit(2, 2)], 2)

ip = np.ones((100, 5))
td = np.ones((100, 2))
a = np.ones((5, 1))
ip.fill(1)
td.fill(10)
gazo = np.random.rand(28, 28) - 0.5
net2.feedforward(gazo)
#def online_bp(self, inputs, t_data, l_rate1, method, drop_out_rate = None, regularization = None, weight_limit = None, l_rate2 = None):
#net.online_bp(ip, td, l_rate1 = 0.02, l_rate2 = 0.0, method = net.GD, do_rate = 0.9, reg_rate = 0.0)
#net.feedforward(a)



