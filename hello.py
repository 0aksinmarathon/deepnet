import numpy as np
from tqdm import tqdm

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

        for i in tqdm(range(len(inputs))):

            input = inputs[i]
            correct = t_data[i]
            self.feedforward(input, do_rate)
            prev_layer = None
            for layer in reversed(self.layers):#逆伝播 後ろの層が何か今の層は何かによって層の種類*層の種類のパターンが生じる
                #print(type(layer))
                #伝播フェーズ

                if isinstance(prev_layer, type(None)):
                    layer.delta = layer.output - np.array(correct).T



                if isinstance(prev_layer, Aff):
                    layer.delta = np.array(layer.activation(layer.output, prime=True)).T * np.matmul(prev_layer.w.T, prev_layer.delta).T

                if isinstance(prev_layer, ReLU):
                    layer.delta = (layer.inflow > 0) * prev_layer.delta

                if isinstance(prev_layer, Pool) and prev_layer.pool_method == 'max':
                    layer.delta = np.zeros_like(layer.output)
                    for k in range(int(prev_layer.input_w / prev_layer.filter_w)):
                        for j in range(int(prev_layer.input_h / prev_layer.filter_h)):
                            sub_mat = prev_layer.input[prev_layer.filter_h * j: prev_layer.filter_h * (j + 1), prev_layer.filter_w * k: prev_layer.filter_w * (k + 1)]
                            layer.delta[prev_layer.filter_h * j: prev_layer.filter_h * (j + 1), prev_layer.filter_w * k: prev_layer.filter_w * (k + 1)] += \
                            (sub_mat == np.max(sub_mat)) * prev_layer.delta[j, k]

                if isinstance(prev_layer, CVL2D):
                    fprime = layer.activation(layer.inflow.reshape((prev_layer.input_h, prev_layer.input_w)), prime = True)
                    a = np.zeros_like(prev_layer.input)
                    for k in range(int(prev_layer.output_w)):
                        for j in range(int(prev_layer.output_h)):
                            a[j: j + prev_layer.filter_h, k: k + prev_layer.filter_w] += \
                                np.array(prev_layer.w * prev_layer.input[j * prev_layer.stride:j * prev_layer.stride + prev_layer.filter_h, k * prev_layer.stride:k * prev_layer.stride + prev_layer.filter_w])
                    layer.delta = fprime * a



                #最適化フェーズ
                if isinstance(layer, Aff):
                    layer.dif = l_rate * np.matmul(layer.delta.reshape(layer.delta.shape[1] * layer.delta.shape[0], 1), layer.input.reshape(1, len(layer.input))) + \
                                reg_rate * np.linalg.norm(layer.w)
                    layer.w = layer.w - layer.dif
                    layer.delta = layer.delta.reshape((layer.delta.shape[0] * layer.delta.shape[1], 1))
                    prev_layer = layer
                    # print(layer.w)
                    # for layer in self.layers:
                    #    layer.w = layer.w * layer.dropout + layer.dropped_out
                if isinstance(layer, ReLU):
                    prev_layer = layer
                    layer.delta = layer.delta.reshape((layer.input_h, layer.input_w))
                if isinstance(layer, Pool):
                    prev_layer = layer

                if isinstance(layer, CVL2D):
                    for j in range(layer.filter_h):
                        for k in range(layer.filter_w):
                            a = layer.input[j: j + layer.output.shape[0], k: k + layer.output.shape[1]] * layer.w[j, k]
                            layer.w[j, k] = layer.w[j, k] - (l_rate * a).sum()
                    prev_layer = layer
                #print(layer.delta)


                    
                    

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



class Aff:
    def __init__(self, input_n, output_n, activation):
        self.input_n = input_n
        self.output_n = output_n
        self.w = np.random.rand(self.output_n, self.input_n)
        self.b = np.random.rand()
        self.activation = activation
    def calc(self, input):
        self.input = input
        self.inflow = np.matmul(self.w, self.input) + self.b

        self.output = self.activation(self.inflow)


class CVL2D:
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
        self.b = np.random.rand()
        self.pad = pad
        self.pad_method = pad_method
        self.w = np.random.rand(filter_h, filter_w)
        self.activation = activation

    def calc(self, input):
        self.input = np.pad(input.reshape(self.input_h, self.input_w), self.pad, self.pad_method)

        self.inflow = np.array([[(self.w * self.input[j * self.stride:j * self.stride + self.filter_h, \
                                             i * self.stride:i * self.stride + self.filter_w]).sum() \
                        for i in range(int(self.output_w))] for j in range(int(self.output_h))])
        self.output = self.activation(self.inflow) + self.b

class CVL3D:
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

class Pool:
    def __init__(self, input_w, input_h, filter_w, filter_h, activation, method = 'max'):
        self.input_w = input_w
        self.input_h = input_h
        self.input_n = input_h * input_w
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.pool_method = method
        self.activation = activation
    def calc(self, input):
        self.input = input
        if self.pool_method == 'max':

            return self.max(input)
        elif self.pool_method == 'mean':
            return self.mean(input)
        elif self.pool_method == 'min':
            return self.mean(input)



    def max(self, input):
        self.output = np.array([[np.max(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                        self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])
        self.inflow = self.output

    def mean(self, input):
        self.output = np.array([[np.mean(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                        self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])
        self.inflow = self.output

    def min(self, input):
        self.output = np.array([[np.min(self.input[self.filter_w * j: self.filter_w * (j + 1), \
                                         self.filter_h * i: self.filter_h * (i + 1)]) \
                                 for i in range(int(self.input_w / self.filter_w))] for j in
                                range(int(self.input_h / self.filter_h))])
        self.inflow = self.output

class ReLU:
    def __init__(self, input_w, input_h, activation):
        self.input_w = input_w
        self.input_h = input_h
        self.activation = activation
    def calc(self, input):
        self.input = input
        negative_to_zero = (input >= 0)
        self.output = (self.input * negative_to_zero).reshape(self.input_w * self.input_h, 1)




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






net = Network([Aff(5, 3, sigmoid),
               Aff(3, 2, sigmoid),
               Aff(2, 784, sigmoid),
               Aff(784, 28, sigmoid),
               Aff(28, 20, sigmoid),
               Aff(20, 2, sigmoid),
               Aff(2, 2, identity)], 1)

net2 = Network([CVL2D(28, 28, sigmoid, 23, 23, stride=1),
                Pool(6, 6, 3, 3, identity, 'max'),
                ReLU(2, 2, activation = identity),
                Aff(4, 20, sigmoid),
                Aff(20, 30, sigmoid),
                Aff(30, 2, identity)
                ],
               2)


ip = np.ones((100, 2))
td = np.ones((100,  1, 2)) + 6
td2 = np.ones((100, 1, 2)) + 20
a = np.ones((28, 28)) + 6
gazo = np.ones((100, 28, 28)) - 0.5
gazo2 = np.ones((100, 28, 28)) + 10
#Affスタートならたてベクトルを入れる
#

#def __init__(self, input_w, input_h):
#net2.feedforward(gazo)

net2.online_bp(np.concatenate((gazo, gazo2)), np.concatenate((td, td2)), method = net2.GD, l_rate1 = 0.001, do_rate = 1, reg_rate = 0.0, weight_limit = 0.0, l_rate2 = 0.0)
#def online_bp(self, inputs, t_data, l_rate1, method, drop_out_rate = None, regularization = None, weight_limit = None, l_rate2 = None):
#net.online_bp(ip, td, l_rate1 = 0.02, l_rate2 = 0.0, method = net.GD, do_rate = 0.9, reg_rate = 0.0)
net2.feedforward(a)



