import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

import numpy as np
import lqr_control as control
import matplotlib.pyplot as plt
from VFA_Net import NeuralNetwork

nn_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "quadratic"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

model = NeuralNetwork(nn_arq, bias=True, double=False)


A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R = np.array(1).reshape(1,1)

x0 = np.array(-1).reshape(1,1)
u0 = np.array(0).reshape(1,1)

# number of time steps to simulate
T = 30
# number of iterations of the dynamical systems for training
NUM_TRIALS = 250
ALPHA = 100
GAMMA = 0.9

K, _, _ = control.dlqr(A,B,Q,R)

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

def train(K):
    loss_history = []
    for i in range(NUM_TRIALS):
        x = np.random.randn(1).reshape(1,1)

            #print('yhat = '+str(y_hat))

        total_loss = 0
        for t in range(T):
            u = -np.matmul(K,x)

            r = np.matmul(x,np.matmul(Q,x)) + np.matmul(u,np.matmul(R,u))

            y = r + ALPHA*GAMMA*model(np.matmul(A,x) + np.matmul(B,u))

            y_hat = model.net_forward(x)

            lr = 0.001

            total_loss += loss(y, y_hat, ALPHA)
            model.net_backward(y, y_hat, ALPHA)
            model.update_wb(lr)

            x = np.matmul(A,x) + np.matmul(B,u)

        #output
        if (i+1)%(NUM_TRIALS/10) == 0 or i == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS))
            print("y =    "+str(model(np.array(1).reshape(1,1))))
            print("u =    "+str(u))
            print("r =    "+str(r))
            print("x+=    "+str(np.matmul(A,x) + np.matmul(B,u)))

        loss_history.append(total_loss/T)
    return loss_history

def live_train(K, low=-1, high=1):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(low,high)
    xtest = np.linspace(low,high,100)

    for j in range(NUM_TRIALS):
        x = np.random.randn(1).reshape(1,1)
        for i in range(T):
            if (i+1)%(T/5) == 0:
                y_hat = np.array([ALPHA*model(np.array(x1).reshape(1,1)).item() for x1 in xtest])
                xs = xtest.reshape(xtest.size,1)
                y = control.trueloss(A,B,Q,R,K,xs,T,GAMMA).reshape(xtest.size)

                ax.clear()
                ax.plot(xtest, y_hat, 'r-')
                ax.plot(xtest, y, 'k-')

                plt.grid(True)
                ax.set_xlabel('x',fontsize=18)
                ax.set_ylabel('y',fontsize=18)
                plt.pause(0.05)

            u = -np.matmul(K,x)
            r = np.matmul(x,np.matmul(Q,x)) + np.matmul(u,np.matmul(R,u))

            y = r + ALPHA*GAMMA*model(np.matmul(A,x) + np.matmul(B,u))
            y_hat = model.net_forward(x)

            lr = 0.001

            model.net_backward(y, y_hat, ALPHA)
            model.update_wb(lr)

            x = np.matmul(A,x) + np.matmul(B,u)

    plt.show()
    return

#print("y_0 =     "+str(model(np.array(1).reshape(1,1))))

#loss_hist = train(K)
#control.plot_V(model,A,B,Q,R,K,T,GAMMA,ALPHA,low=-3,high=3)

live_train(K)
