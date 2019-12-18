import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

import numpy as np
from matplotlib import pyplot as plt
from VFA_Net import NeuralNetwork

nn_arq = [
    {"input_dim": 1, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "none"},
]

ALPHA = 1
NUM_SAMPLES = 10000
BATCH_SIZE = 1

## slope and offset of function to be learnied
AA = 0.7
BB = 0.3

def f_star(x):
    return AA * x + BB

model = NeuralNetwork(nn_arq, bias=True, double=True, zero=False, seed=1)

def process(x):
    return np.array(x).reshape(1,1)

def sample_train(nsample):
    xtrain = np.random.randn(nsample).reshape(nsample,1)
    return xtrain, f_star(xtrain)

def live_train(xtrain, epochs=10, low=-1, high=1, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(low,high)
    xtest = np.linspace(low,high,100)
    
    for j in range(epochs):
        if (j+1)%(epochs/10) == 0:
            print('epoch: {}/{}'.format(j+1,epochs))
        
        for i in range(NUM_SAMPLES):
            if (i+1)%(NUM_SAMPLES/100) == 0:
                y_hat = np.array([ALPHA*model(process(x)).item() for x in xtest])
                y = np.array([f_star(x) for x in xtest])
    
                ax.clear()
                ax.plot(xtest, y_hat, 'r-')
                ax.plot(xtest, y, 'k-')
        
            
                plt.grid(True)
                plt.pause(0.05)
            
            for k in range(BATCH_SIZE):

                y_hat = model.net_forward(process(xtrain[i]))
                y = np.array(f_star(xtrain[i])).reshape(1,1)

                lr = 0.001

                model.net_backward(y, y_hat, ALPHA)
                model.update_wb(lr)
        
    plt.show()
    return

xtrain,ytrain = sample_train(NUM_SAMPLES)
live_train(xtrain, epochs=1, low=-2, high=2)