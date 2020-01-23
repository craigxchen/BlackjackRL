import numpy as np
from collections import defaultdict

class PGNet:
    def __init__(self, nn_structure, bias=True, double=False, zero=False, seed=None, initVar = 1, initVarLast = 1):
        self.nn_structure = nn_structure
        self.num_layers = len(nn_structure)
        self.parameters = {}
        self.bias = bias
        self.double = double
        self.zero = zero
        # Variance of all layers' parameters minus last
        self.initVar = initVar
        # Variance of last layer parameters
        self.initVarLast = initVarLast

        # intializes dictionaries needed to store values for backpropagation
        self.memory = {}
        self.grad_values = {}

        # TODO: add if statement to catch when double and zero are both true

        if seed is not None:
            np.random.seed(seed)

        # loop through neural net
        for idx, layer in enumerate(self.nn_structure):

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            if self.bias:
                self.parameters['b_' + str(idx)] = np.random.randn(layer_output_size, 1)
            else:
                self.parameters['b_' + str(idx)] = np.zeros((layer_output_size,1))

            self.parameters['w_' + str(idx)] = np.random.normal(0, self.initVarLast/layer_input_size, (layer_output_size, layer_input_size))

            if self.double and idx == self.num_layers-1:
                if layer_input_size%2 != 0:
                    raise Exception('Odd number of layers in the last layer, must be even to use doubling trick')

                # TODO: change so it doubles every hidden layer to ensure initialization to 0 (?)
                for i in range(layer_output_size):
                    halfArray = np.random.normal(0, self.initVarLast/layer_input_size, int(layer_input_size/2))
                    self.parameters['w_' + str(idx)][i] = np.concatenate((halfArray,np.negative(halfArray)))

            # sets weights of last layer to 0
            elif self.zero  and idx == self.num_layers-1:
                self.parameters['w_' + str(idx)] = np.zeros((layer_output_size,layer_input_size))


    def __call__(self, a0):
        # TODO: add assertion to confirm shape of input
        a_prev = a0
        for idx, layer in enumerate(self.nn_structure):
            w_n = self.parameters['w_' + str(idx)]
            b_n = self.parameters['b_' + str(idx)]

            a_n, z_n = self.layer_activation(a_prev, w_n, b_n, layer['activation'])
            a_prev = a_n

        return a_n

    def layer_activation(self, a_prev, w, b, activation = 'relu'):
        # function computes the process that occurs in a single layer
        # returns the activation value and the z value, both are needed for the gradient
        z = np.matmul(w,a_prev) + b
        if activation == 'none':
            return z, z
        elif activation == 'relu':
            return self.relu(z), z
        elif activation == 'softmax':
            return self.softmax(z), z
        else:
            raise Exception('activation function currently not supported')

    def net_forward(self, a0):
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

    def gradient_backward(self, a_prev, w_n, z_n, dA, activation = 'relu'):
        if activation == 'none':
            dZ = dA
        elif activation == 'relu':
            dZ = dA * self.drelu(z_n)
        elif activation == 'softmax':
            dZ = np.matmul(self.dsoftmax(z_n),dA)
        else:
            raise Exception('activation function currently not supported')

        dA_prev = np.matmul(w_n.T, dZ)
        dW = np.matmul(dZ, a_prev.T)
        if self.bias:
            dB = dZ
        else:
            dB = np.zeros(dZ.shape)

        return dA_prev, dW, dB

    """
    The following functions compute the backward pass for the gradient ascent of the policy gradient routine
    the derivative of the cost function varies depending on the parametrization of
    the actor network, stored in the 'criticstyle' variable:
    """

    def net_backward_SPG_tabular(self, targets, predictions):
        """
        SPG_tabular = stochastic policy gradient (SPG) with tabular
                      critic, cross entropy loss
        """
        #TODO: change code to allow for multidimensional actions
        # cross entropy loss derivativeS
        dA = targets/predictions

        for idx, layer in reversed(list(enumerate(self.nn_structure))):
            if idx == 0:
                a_prev = self.input_batch
            else:
                a_prev = self.memory['a_' + str(idx-1)]

            z_n = self.memory['z_' + str(idx)]
            w_n = self.parameters['w_' + str(idx)]

            dA_prev, dW, dB = self.gradient_backward(a_prev, w_n, z_n, dA, layer['activation'])

            dA = dA_prev

            self.grad_values['dW_' + str(idx)] = dW
            self.grad_values['dB_' + str(idx)] = dB

        return self.grad_values


    def net_backward_SPG_normal(self, advantages, predictions, actions, SIGMA):
        """
        SPG with gaussian distro, the neural network parametrizes the mean while the variance is kept constant (SIGMA)
        advantages = samples of the advantage function
        actions = samples of actions during the sampled MDP
        predictions = outputs of neural network for sequence of states from sampled MDP, i.e.
                      values of the average of the stochastic policy for those states.
        SIGMA = variance of the stochastic policy
        """
        #TODO: change code to allow for multidimensional actions
        # cross entropy loss derivative times derivative of normal distro wrt mean
        dA = -advantages*(predictions-actions)/SIGMA**2

        for idx, layer in reversed(list(enumerate(self.nn_structure))):
            if idx == 0:
                a_prev = self.input_batch
            else:
                a_prev = self.memory['a_' + str(idx-1)]

            z_n = self.memory['z_' + str(idx)]
            w_n = self.parameters['w_' + str(idx)]

            dA_prev, dW, dB = self.gradient_backward(a_prev, w_n, z_n, dA, layer['activation'])

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
        temp = defaultdict(list)
        for i in range(len(grad_values)):
            for idx, _ in enumerate(self.nn_structure):
                temp['dW_'+str(idx)].append(grad_values[i]['dW_'+str(idx)])
                temp['dB_'+str(idx)].append(grad_values[i]['dB_'+str(idx)])
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['w_' + str(idx)] -= step_size*np.mean(temp['dW_' + str(idx)], axis=0)
            self.parameters['b_' + str(idx)] -= step_size*np.mean(temp['dB_' + str(idx)], axis=0)
        return

    # activation functions
    def relu(self, x):
        return np.maximum(0,x)

    def drelu(self, x):
        return (x > 0).astype(int)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def dsoftmax(self, x):
        s = self.softmax(x)
        return np.diagflat(s) - np.matmul(s, s.T)
