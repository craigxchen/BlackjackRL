import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def lqr(A,B,Q,R):
    '''
    Solves for the optimal infinite-horizon, continuous-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    '''

    S = scipy.linalg.solve_continuous_are(A, B, Q, R)

    K = np.matmul(scipy.linalg.inv(R), np.matmul(B.T, S))

    eigVals, eigVecs = scipy.linalg.eig(A-np.matmul(B,K))

    return K, S, eigVals

def dlqr(A,B,Q,R):
    '''
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    '''

    S = scipy.linalg.solve_discrete_are(A, B, Q, R)

    F = np.matmul(scipy.linalg.inv(np.matmul(np.matmul(B.T, S), B) + R), (np.matmul(np.matmul(B.T, S), A)))

    eigVals, eigVecs = scipy.linalg.eig(A - np.matmul(B, F))

    return F, S, eigVals

def simulate_discrete(A,B,K,x0,u0,T):
    '''
    simulates the linear system (A,B) with static control law
    u(t) = K @ x(t)
    from initial condition x0 for T time steps

    returns matrices u and x of control and state trajectories, respectively.
    rows are indexed by time
    '''
    x = x0
    u = u0
    for t in range(T):
        u_t = np.matmul(-K, x[:,[-1]])
        x_prime = np.matmul(A, x[:,[-1]]) + np.matmul(B, u_t)
        x = np.hstack((x, x_prime))
        u = np.hstack((u, u_t))
    return x, u

def trueloss(A,B,Q,R,K,xs,T,gamma):
    """
    !!!!!not sure that this works in hiher dimension!!!!
    input:
        - all the matrices
        - xs is a np.array of shape (l,1) of initial conditions
        - T is the number of steps
    output:
        - computes the sum of discounted rewards starting from the initial conditions stored in x0s up to time T
    """
    V = np.zeros(xs.size).reshape(xs.shape)
    for j in range(T):
        us = -K*xs
        V += (gamma**j)*(xs*Q*xs + us*R*us)
        xs = A*xs + B*us
    return V

#def trueloss2(A,B,Q,R,K,xs,T):
#    """
#    !!!!!not sure that this works in hiher dimension!!!!
#    input:
#        - all the matrices
#        - xs is a np.array of shape (l,1) of initial conditions
#        - T is the number of steps
#    output:
#        - computes the sum of discounted rewards starting from the initial conditions stored in x0s up to time T
#    """
#    V = np.zeros(x.size).reshape(x.shape)
#    for j in range(T):
#        us = -K*xs
#        V += (gamma**j)*(xs*Q*xs + us*R*us)
#        xs = A*xs + B*us
#    return V


def plot_paths(x1,x2,ylabel,R1,R2):
    fig, ax = plt.subplots()
    colors = [ '#2D328F', '#F15C19' ] # blue, orange
    label_fontsize = 18

    t = np.arange(0,x1.shape[0])
    ax.plot(t,x1,color=colors[0],label='R={}'.format(R1[0][0]))
    ax.plot(t,x2,color=colors[1],label='R={}'.format(R2[0][0]))

    ax.set_xlabel('Time',fontsize=label_fontsize)
    ax.set_ylabel(ylabel,fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

# TODO fix
def plot_states(x,ylabel,R):
    fig, ax = plt.subplots()
    colors = [ '#B53737', '#0B6623', '#2D328F'] # red, green, blue
    label_fontsize = 18

    t = np.arange(0,x.shape[1])
    # change to be a loop in the future that supports N colors
    ax.plot(t,x[0],color=colors[0],label='Node 1')
    ax.plot(t,x[1],color=colors[1],label='Node 2')
    ax.plot(t,x[2],color=colors[2],label='Node 3')

    ax.set_xlabel('Time',fontsize=label_fontsize)
    ax.set_ylabel(ylabel + ' (R={})'.format(R[0][0]),fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

def plot_loss(loss):
    fig, ax = plt.subplots()
    colors = [ '#B53737'] # red
    label_fontsize = 18

    # change to be a loop in the future that supports N colors
    ax.plot(loss,color=colors[0],label='Loss')

    ax.set_xlabel('Time',fontsize=label_fontsize)
    ax.set_ylabel('TD error')
    plt.legend(fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

def plot_V(model,A,B,Q,R,K,Tm,gamma,alpha,low=-1,high=1):
    fig, ax = plt.subplots()
    colors = [ '#B53737', '#2D328F' ] # red, blue
    label_fontsize = 18

    x = np.arange(low,high,0.1)

    ax.plot(x,[alpha*model(np.array(x1).reshape(1,1)).item() for x1 in x],color=colors[0],label='Approx. Loss Function')
    xs = x.reshape(x.size,1)
    ax.plot(x,trueloss(A,B,Q,R,K,xs,Tm,gamma).reshape(x.size),color=colors[1],label='Real Loss Function')


    ax.set_xlabel('x',fontsize=label_fontsize)
    ax.set_ylabel('y',fontsize=label_fontsize)
    plt.legend()

    plt.grid(True)
    plt.show()
    return
