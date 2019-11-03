import numpy as np

class NeuralNetwork:
    def __init__(self, nn_structure, double="no", loss=None):
        self.nn_structure = nn_structure
        self.num_layers = len(nn_structure)
        
        self.parameters = {}
        
        # intializes dictionaries needed to store values for backpropagation
        self.memory = {}
        self.grad_values = {}
        
        for idx, layer in enumerate(self.nn_structure):
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"] 
            # using He intialization for ReLU
#            # np.tile to create 3D array
#            self.parameters['w_' + str(idx)] = np.tile(np.random.randn(layer_output_size, layer_input_size)
#                                , (self.batch_size, 1, 1)) * np.sqrt(2/layer_input_size)                                            
#            self.parameters['b_' + str(idx)] = np.tile(np.random.randn(layer_output_size, 1) * 0.1
#                                , (self.batch_size, 1, 1))
            if double == "yes":
                temp_w = (np.random.random(size=layer_output_size*layer_input_size)*np.sqrt(1/layer_input_size)).tolist()
                dbl_w = [k*(-1**i) for k,i in zip(temp_w,range(len(temp_w)))]
                
                temp_b = (np.random.random(size=layer_output_size)*np.sqrt(1/layer_input_size)).tolist()
                dbl_b = [k*(-1**i) for k,i in zip(temp_b,range(len(temp_b)))]
                
                self.parameters['w_' + str(idx)] = np.array(dbl_w).reshape(layer_output_size, layer_input_size)
                self.parameters['b_' + str(idx)] = np.array(dbl_b).reshape(layer_output_size, 1)
            else:
                self.parameters['w_' + str(idx)] = np.random.randn(layer_output_size, layer_input_size)
                self.parameters['b_' + str(idx)] = np.random.randn(layer_output_size, 1)
            
    def __call__(self, a0):
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
        elif activation == 'sigmoid':
            return self.sigmoid(z), z
        elif activation == 'tanh':
            return self.tanh(z), z
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

    def gradient_backward(self, a_n, a_prev, w_n, z_n, dA, activation = 'relu'):
        if activation == 'none':
            dZ = dA
        elif activation == 'relu':
            dZ = dA * self.drelu(z_n)
        elif activation == 'sigmoid':
            dZ = dA * self.dsigmoid(z_n)
        elif activation == 'tanh':
            dZ = dA * self.dtanh(z_n)
        else:
            raise Exception('activation function currently not supported')
        
        dA_prev = np.matmul(w_n.T, dZ)
        dW = np.matmul(dZ, a_prev.T)
        dB = dZ
        
        return dA_prev, dW, dB 
    
    def net_backward(self, targets, predictions, alpha=1):
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
            
        return
    
    def update_wb(self, step_size):
        for idx, layer in enumerate(self.nn_structure):
            self.parameters['w_' + str(idx)] -= step_size*self.grad_values['dW_' + str(idx)] 
            self.parameters['b_' + str(idx)] -= step_size*self.grad_values['dB_' + str(idx)]
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
    
    def reset_params(self):
        self.parameters = {}
        self.memory = {}
        self.grad_values = {}
        for idx, layer in enumerate(self.nn_structure):
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"] 
            
            self.parameters['w_' + str(idx)] = np.random.randn(layer_output_size, layer_input_size) * np.sqrt(1/layer_input_size)
            self.parameters['b_' + str(idx)] = np.random.randn(layer_output_size, 1) * 0.1
        return
    
    # activation functions
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0,x)
    
    def dsigmoid(self, x):
        return np.multiply(self.sigmoid(x), np.ones(x.shape) - self.sigmoid(x))
    
    def drelu(self, x):
        return (x > 0).astype(int)
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        
    def dtanh(self, x):
        return 1 - self.tanh(x)**2
