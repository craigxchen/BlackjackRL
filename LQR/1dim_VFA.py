import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

import numpy as np
from lqr_control import control
from VFA_Net import NeuralNetwork

nn_arq = [
    {"input_dim": 1, "output_dim": 64, "activation": "quadratic"},
    {"input_dim": 64, "output_dim": 1, "activation": "none"},
]

model = NeuralNetwork(nn_arq, bias=False, double=True)

A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R1 = np.array(1).reshape(1,1)
R2 = np.array(1.2).reshape(1,1)

x0 = np.array(-1).reshape(1,1)
u0 = np.array(0).reshape(1,1)

# number of time steps to simulate
T = 30
# number of iterations of the dynamical systems for training
NUM_TRIALS = 1000
ALPHA = 100
GAMMA = 0.9

K_1, _, _ = control.dlqr(A,B,Q,R1)
x_1, u_1 = control.simulate_discrete(A,B,K_1,x0,u0,T)

K_2, _, _ = control.dlqr(A,B,Q,R2)
x_2, u_2 = control.simulate_discrete(A,B,K_2,x0,u0,T)

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

            r = np.matmul(x,np.matmul(Q,x)) + np.matmul(u,np.matmul(R1,u))

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
    return  loss_history

print("y_0 =     "+str(model(np.array(1).reshape(1,1))))

loss_hist = train(K_1)

control.plot_loss(loss_hist)

control.plot_V(model)

control.plot_paths(x_1[0],x_2[0],'Position',R1,R2)
