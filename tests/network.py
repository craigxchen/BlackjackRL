import numpy as np
from collections import defaultdict

class NeuralNetwork:
    def __init__(self, nn_structure, bias=True, double=False, seed=None, initVar = 1,initVarLast = 1):
        self.nn_structure = nn_structure
        self.num_layers = len(nn_structure)
        self.parameters = {}
        self.bias = bias
        self.double = double
        self.initVar = initVar
        self.initVarLast = initVarLast

        # intializes dictionaries needed to store values for backpropagation
        self.memory = {}
        self.grad_values = {}

        if seed is not None:
            np.random.seed(seed)
        for idx, layer in enumerate(self.nn_structure):

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            if self.bias:
                self.parameters['b_' + str(idx)] = np.random.randn(layer_output_size, 1)
            else:
                self.parameters['b_' + str(idx)] = np.zeros((layer_output_size,1))

            self.parameters['w_' + str(idx)] = np.random.normal(0, self.initVar/layer_input_size**2, (layer_output_size, layer_input_size))

            if self.double and idx == self.num_layers-1:
                if layer_input_size%2 != 0:
                    raise Exception('Odd number of layers in the last layer, must be even to use doubling trick')
                # sets weights of last layer to 0
                if initVarLast != 0:
                    for i in range(layer_output_size):
                        halfArray = np.random.normal(0, self.initVarLast/layer_input_size**2, int(layer_input_size/2))
                        self.parameters['w_' + str(idx)][i] = np.concatenate((halfArray,np.negative(halfArray)))
                else:
                    self.parameters['w_' + str(idx)] = np.zeros((layer_output_size,layer_input_size))


    def __call__(self, a0):
        a_prev = a0
        for idx, layer in enumerate(self.nn_structure):
            w_n = self.parameters['w_' + str(idx)]
            b_n = self.parameters['b_' + str(idx)]

            a_n, z_n = self.layer_activation(a_prev, w_n, b_n, layer['activation'])
            a_prev = a_n

        return a_n

    def layer_activation(self, a_prev, w, b, activation = 'relu'):
        "inputs: a_prev nx1 vector, w nxm matrix, b 1xm vector "
        # function computes the process that occurs in a single layer
        # returns the activation value and the z value, both are needed for the gradient

        z = np.matmul(w,a_prev) +b
        if activation == 'none':
            return z, z
        elif activation == 'relu':
            return self.relu(z), z
        elif activation == 'sigmoid':
            return self.sigmoid(z), z
        elif activation == 'tanh':
            return self.tanh(z), z
        elif activation == 'leakyRelu':
            return self.leakyRelu(z), z
        elif activation == 'quadratic':
            return self.quadratic(z), z
        else:
            raise Exception('activation function currently not supported')

    def net_forward(self, a0):#
        self.input_batch = a0
        a_prev = a0
        for idx, layer in enumerate(self.nn_structure):
            w_n = self.parameters['w_' + str(idx)]
            b_n = self.parameters['b_' + str(idx)]


            a_n, z_n = self.layer_activation(a_prev, w_n, b_n, layer['activation'])
            a_prev = a_n

            self.memory['a_' + str(idx)] = a_n
            self.memory['z_' + str(idx)] = z_n
        return a_n

    def gradient_backward(self, a_n, a_prev, w_n, z_n, dA, activation = 'relu'):
        if activation == 'none':
            dZ = dA
        elif activation == 'relu':
            dZ = dA * self.drelu(z_n)
        elif activation == 'sigmoid':
            dZ = dA * self.dsigmoid(z_n)
        elif activation == 'tanh':
            dZ = dA * self.dtanh(z_n)
        elif activation == 'leakyRelu':
            dZ = dA * self.dleakyRelu(z_n)
        elif activation == 'quadratic':
            dZ = dA * self.dquadratic(z_n)
        else:
            raise Exception('activation function currently not supported')

        dA_prev = np.matmul(w_n.T, dZ)
        #print("a:_____"+str(a_prev))
        dW = np.matmul(dZ, a_prev.T)
        if self.bias:
            dB = dZ
        else:
            dB = np.zeros(dZ.shape)

        return dA_prev, dW, dB

    def net_backward(self, targets, predictions, alpha=1): #what are the inputs here?
        # derivative of cost w.r.t. final activation (1/alpha^2 MSE)
        dA = -(1/alpha)*(targets - alpha*predictions)
        for idx, layer in reversed(list(enumerate(self.nn_structure))):
            if idx == 0:

                a_prev = self.input_batch
            else:
                a_prev = self.memory['a_' + str(idx - 1)]

            z_n = self.memory['z_' + str(idx)]
            w_n = self.parameters['w_' + str(idx)]

            dA_prev, dW, dB = self.gradient_backward(predictions, a_prev, w_n, z_n, dA, layer['activation'])

            dA = dA_prev

            self.grad_values['dW_' + str(idx)] = dW
            self.grad_values['dB_' + str(idx)] = dB

        return self.grad_values

    def update_wb(self, step_size):
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['w_' + str(idx)] -= step_size*self.grad_values['dW_' + str(idx)]
            self.parameters['b_' + str(idx)] -= step_size*self.grad_values['dB_' + str(idx)]
        return

    def batch_update_wb(self, step_size, grad_values):
        temp = defaultdict(lambda: [])
        print("------------------------------\n")

        for i in range(len(grad_values)):
            for idx, _ in enumerate(self.nn_structure):
                temp['dW_'+str(idx)].append(grad_values[i]['dW_'+str(idx)])
                temp['dB_'+str(idx)].append(grad_values[i]['dB_'+str(idx)])
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['w_' + str(idx)] -= step_size*np.mean(temp['dW_' + str(idx)], axis=0)
            self.parameters['b_' + str(idx)] -= step_size*np.mean(temp['dB_' + str(idx)], axis=0)
        return

    def save_model(self, name):
        np.save(name, self.parameters)
        return

    def load_model(self, name):
        npfile = np.load('{}.npy'.format(name), allow_pickle=True).item()
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['w_' + str(idx)] = npfile['w_' + str(idx)]
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['b_' + str(idx)] = npfile['b_' + str(idx)]
        return

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        for idx, layer in enumerate(self.nn_structure):

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            if self.bias:
                self.parameters['b_' + str(idx)] = np.random.randn(layer_output_size, 1)
            else:
                self.parameters['b_' + str(idx)] = np.zeros((layer_output_size,1))

            self.parameters['w_' + str(idx)] = np.random.normal(0, self.initVar/layer_input_size**2, (layer_output_size, layer_input_size))

            if self.double and idx == self.num_layers-1:
                if layer_input_size%2 != 0:
                    raise Exception('Odd number of layers in the last layer, must be even to use doubling trick')
                # sets weights of last layer to 0
                if initVarLast != 0:
                    for i in range(layer_output_size):
                        halfArray = np.random.normal(0, self.initVarLast/layer_input_size**2, int(layer_input_size/2))
                        self.parameters['w_' + str(idx)][i] = np.concatenate((halfArray,np.negative(halfArray)))
                else:
                    self.parameters['w_' + str(idx)] = np.zeros((layer_output_size,layer_input_size))

    # activation functions
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def dsigmoid(self, x):
        return np.multiply(self.sigmoid(x), np.ones(x.shape) - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0,x)

    def drelu(self, x):
        return (x > 0).astype(int)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def dtanh(self, x):
        return 1 - self.tanh(x)**2

    def leakyRelu(self, x, a=0.2):
        return np.maximum(a*x, x)

    def dleakyRelu(self, x, a=0.2):
        return (x > 0).astype(int) - a*(x < 0).astype(int)

    def quadratic(self, x):
        return np.square(x)

    def dquadratic(self, x):
        return 2*x
