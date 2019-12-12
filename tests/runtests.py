import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

from VFA_Net import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

nn_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

ALPHA = 1
NUM_SAMPLES = 20000
BATCH_SIZE = 1
#TMAX = 20000 seems not needed? does the same thing as NUM_SAMPLES - I've replaced it in the implementation, you can always revert that

## slope and offset of function to be learnied
AA = 0.7
BB = 0.3

def f_star(x):
    return AA * x + BB

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

model = NeuralNetwork(nn_arq, bias=True, double=True, initVar=1, initVarLast=1)


# perhaps this might make things easier
#AA: Nice, thanks!
def process(x):
    return np.array(x).reshape(1,1)


def sample_train(nsample):
    xtrain = np.random.randn(nsample).reshape(nsample,1)
    return xtrain, f_star(xtrain) #xtrain, ytrain

def plot_loss(y):
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(t[::int(NUM_SAMPLES/50)],y[::int(NUM_SAMPLES/50)])

    ax.set_yscale('log')
    ax.set_xlabel('Trials',fontsize=label_fontsize)
    ax.set_ylabel('Loss',fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

# broken!
def plot_test(xtest): # xtest is a np.array
    fig, ax = plt.subplots()
    label_fontsize = 18

    ax.plot(list(xtest),[model(process(x)) for x in xtest], label="model")
    ax.plot(list(xtest),list(f_star(xtest)), label="f*")

    ax.set_xlabel('x',fontsize=label_fontsize)
    ax.set_ylabel('yhat',fontsize=label_fontsize)
    ax.legend()

    plt.grid(True)
    plt.show()
    return

# I suggest we use this to compare the model and f* instead of plot_test - Craig
def comparison(low=-1,high=1):
    # these should all be 1D numpy arrays
    xtest = np.linspace(low,high,1000)
    y_hat = np.array([model(process(x)).item() for x in xtest])
    y = np.array([f_star(x) for x in xtest])

    fig, ax = plt.subplots()
    label_fontsize = 18

    ax.plot(xtest, y_hat, 'r-', label="model")
    ax.plot(xtest, y, 'k-', label="f*")

    ax.set_xlabel('x',fontsize=label_fontsize)
    ax.set_ylabel('yhat',fontsize=label_fontsize)
    ax.legend()

    plt.grid(True)
    plt.show()
    return

# %% training

def train(xtrain, epochs=10, **kwargs):
    loss_history = []

    for j in range(epochs):
        if (j+1)%(epochs/10) == 0:
            print('epoch: {}/{}'.format(j+1,epochs))

#        grad_values = {}
        for i in range(NUM_SAMPLES):
#            if (i+1)%(xtrain.shape[0]/10) == 0:
#                print('training samples {}/{}'.format(i+1,xtrain.shape[0]))
            for k in range(BATCH_SIZE):

                y_hat = model.net_forward(process(xtrain[i]))
                y = process(f_star(xtrain[i]))

                lr = 0.001

                loss_history.append(loss(y, y_hat, ALPHA))
                model.net_backward(y, y_hat, ALPHA)
                model.update_wb(lr)

#                grad_values[i] = model.net_backward(y, y_hat, ALPHA)

#            model.batch_update_wb(lr,grad_values)

    return loss_history

xtrain,ytrain = sample_train(NUM_SAMPLES)
loss_history = train(xtrain, epochs=10)
plot_loss(loss_history)
#plot_test(np.random.randn(50))
comparison()
