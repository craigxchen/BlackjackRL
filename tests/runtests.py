from network import NeuralNetwork
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import numpy as np

# try last-layer zeros or doubling
# lowering step size
# increasing alpha
# try batch update

"""
test results:

    using ALPHA = 1000, GAMMA = 1,
    NUM_TRIALS = 100000 and 500000
    512 neurons in hidden layer

    function initialized to zero

    converges when:
        1-hot encoding and relu, leakyRelu
        normalized vector encoding and _
"""


nn_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

ALPHA = 1
GAMMA = 1
NUM_TRIALS = 500000
TMAX = 20000
#slope and offset of function to be learnied
AA = 0.7
BB = 0.3

def fstar(x):
    return AA * x + BB

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

model = NeuralNetwork(nn_arq, bias = True, double = "yes", initVar = 1, initVarLast = 1)


#


def sample_train(nsample):
    xtrain = np.random.randn(nsample)
    return xtrain, fstar(xtrain) #xtrain, ytrain


def plot_loss(y):
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(t[::int(TMAX/50)],y[::int(TMAX/50)])

    ax.set_yscale('log')
    ax.set_xlabel('Trials',fontsize=label_fontsize)
    ax.set_ylabel('Loss',fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

def plot_test(xtest):# xtest is a np.array
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(list(xtest),[model.net_forward([x]) for x in xtest])
    ax.plot(list(xtest),list(fstar(xtest)))
    #ax.set_yscale('log')
    ax.set_xlabel('x',fontsize=label_fontsize)
    ax.set_ylabel('yhat',fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

# %% training

def train(xtrain, **kwargs):
    loss_history = []

    for j in range(TMAX):
        grad_values = {}
        losstemp = 0
        for i in range(xtrain.shape[0]):
            if (i+1)%(xtrain.shape[0]/10) == 0:
                print('training samples {}/{}'.format(i+1,xtrain.shape[0]))

            y_hat = model.net_forward(np.array([[xtrain[i]]]))
            y = np.array([[fstar(xtrain[i])]])

            lr = 0.001

            losstemp += loss(y, y_hat, ALPHA)
            #model.net_backward(y, y_hat, ALPHA)

            grad_values[i] = model.net_backward(y, y_hat[0], ALPHA)
        model.batch_update_wb(lr,grad_values)
        loss_history.append(losstemp/xtrain.shape[0])

    return loss_history

#model.reset_params()
xtrain,ytrain = sample_train(500)
loss_history = train(xtrain)
plot_loss(loss_history)
plot_test(np.random.randn(50))
