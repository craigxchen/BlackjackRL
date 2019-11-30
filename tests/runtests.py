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

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

model = NeuralNetwork(nn_arq, bias = True, double = "yes", initVar = 1, initVarLast = 1)


# %%



def sample_train(nsample):
    xtrain = np.random.randn(nsample)
    return xtrain, 0.7*xtrain + 1.2 #xtrain, ytrain


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

def plot_test(xtest):
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(xtest,[model.net_forward(x) for x in xtest])
    ax.plot(xtest,[0.7*x+1.2 for x in xtest])
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
        for i in range(xtrain.shape):
            if (i+1)%(xtrain.shape/10) == 0:
                print('trial {}/{}'.format(i+1,xtrain.shape))

            y_hat = model.net_forward(process(state))

            lr = 0.001

            losstemp += loss(y, y_hat, ALPHA)
            model.net_backward(y, y_hat, ALPHA)

            grad_values[str(i)] = model.net_backward(y, y_hat, ALPHA)
        model.batch_update_wb(lr,grad_values)
        loss_history.append(losstemp/xtrain.shape)

    return loss_history

#model.reset_params()
xtrain,ytrain = sample_train(500)
loss_history = train(xtrain)
plot_loss(loss_history)
plot_test(list(np.random.randn(50)))
