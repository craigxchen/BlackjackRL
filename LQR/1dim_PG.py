import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

import numpy as np
import lqr_control as control
import matplotlib.pyplot as plt
from VFA_Net import NeuralNetwork
from PG_Net import PGNet

ctc_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "quadratic"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

act_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "quadratic"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

ACTOR = PGNet(act_arq, bias=True, double=True)
CRITIC = NeuralNetwork(ctc_arq, bias=False, double=True)

SIGMA = 0.01


A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R = np.array(1).reshape(1,1)

x0 = np.array(-1).reshape(1,1)
u0 = np.array(0).reshape(1,1)

# number of time steps to simulate for critic
T_CTC = 30
# number of iterations of the dynamical systems for training for critic network
NUM_TRIALS_CTC = 100
ALPHA_CTC = 100
GAMMA_CTC = 0.9

# number of time steps to simulate for critic
T_ACT = 30
# number of iterations of the dynamical systems for training for critic network
NUM_TRIALS_ACT = 100
ALPHA_ACT = 100
GAMMA_ACT = 0.9

K, _, _ = control.dlqr(A,B,Q,R)

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

def train(K):
    loss_history = []
    for i in range(NUM_TRIALS):
        x = np.random.randn(1).reshape(1,1)

            #print('yhat = '+str(y_hat))

        total_loss = 0
        for t in range(T_CTC):
            mu = CRITIC.net_forward(x) # return action suggested by critic

            u_samp = np.random.normal(mu, SIGMA)

            r = np.matmul(x,np.matmul(Q,x)) + np.matmul(u,np.matmul(R,u))

            y = r + ALPHA_CTC*GAMMA_CTC*CRITIC(np.matmul(A,x) + np.matmul(B,u))

            y_hat = CRITIC.net_forward(x)

            lr_CTC = 0.001

            total_loss_CTC += -np.matmul(K,x)-mu
            CRITIC.net_backward(y, y_hat, ALPHA_CTC)
            CRITIC.update_wb(lr_CTC)

            lr_ACT = 0.0001
            advantage = r+ALPHA_CTC*CRITIC(np.matmul(A,x) + np.matmul(B,u))-y_hat

            total_loss_ACT += loss(y, y_hat)
            ACTOR.net_backward_SPG_normal(advantage, mu, u_samp, SIGMA)
            ACTOR.update_wb(lr_ACT)

            x = np.matmul(A,x) + np.matmul(B,u)

        #output
        if (i+1)%(NUM_TRIALS/10) == 0 or i == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS_CTC))
            print("y =    "+str(model(np.array(1).reshape(1,1))))
            print("u =    "+str(u))
            print("r =    "+str(r))
            print("x+=    "+str(np.matmul(A,x) + np.matmul(B,u)))

        loss_history.append(total_loss/T_CTC)
    return loss_history

def live_train(K, low=-1, high=1):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(low,high)
    xtest = np.linspace(low,high,100)

    for j in range(NUM_TRIALS_CTC):
        x = np.random.randn(1).reshape(1,1)
        for i in range(T_CTC):
            if (i+1)%(T_CTC/5) == 0:
                y_hat = np.array([ALPHA_CTC*model(np.array(x1).reshape(1,1)).item() for x1 in xtest])
                xs = xtest.reshape(xtest.size,1)
                y = control.trueloss(A,B,Q,R,K,xs,T_CTC,GAMMA_CTC).reshape(xtest.size)

                ax.clear()
                ax.plot(xtest, y_hat, 'r-')
                ax.plot(xtest, y, 'k-')

                plt.grid(True)
                ax.set_xlabel('x',fontsize=18)
                ax.set_ylabel('y',fontsize=18)
                plt.pause(0.05)

            u = -np.matmul(K,x)
            r = np.matmul(x,np.matmul(Q,x)) + np.matmul(u,np.matmul(R,u))

            y = r + ALPHA_CTC*GAMMA_CTC*model(np.matmul(A,x) + np.matmul(B,u))
            y_hat = model.net_forward(x)

            lr = 0.001

            model.net_backward(y, y_hat, ALPHA_CTC)
            model.update_wb(lr)

            x = np.matmul(A,x) + np.matmul(B,u)

    plt.show()
    return

#print("y_0 =     "+str(model(np.array(1).reshape(1,1))))

#loss_hist = train(K)
#control.plot_V(model,A,B,Q,R,K,T,GAMMA,ALPHA,low=-3,high=3)

live_train(K)
