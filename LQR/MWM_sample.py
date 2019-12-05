import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R):
    '''
    # Solves for the optimal infinite-horizon discrete-time LQR cost matrix given linear system (A,B) 
    # and cost function parameterized by (Q,R)
    '''
    
    M=scipy.linalg.solve_discrete_are(A,B,Q,R)

    # K=(B'MB + R)^(-1)*(B'MA)
    return np.dot(scipy.linalg.inv(np.matmul(np.matmul(B.T,M),B)+R),(np.matmul(np.matmul(B.T,M),A)))

def simulate(A,B,K,x0,T):
    '''
    simulates the linear system (A,B) with static control law
    u(t) = K @ x(t)
    from initial condition x0 for T time steps

    returns matrices u and x of control and state trajectories, respectively.
    rows are indexed by time
    '''
    x = x0
    u = np.array(0).reshape(1,1) #init to 0
    for t in range(T):
        u_t = np.matmul(-K, x[:,[-1]])
        x_prime = np.matmul(A, x[:,[-1]]) + np.matmul(B, u_t)
        x = np.hstack((x, x_prime))
        u = np.hstack((u, u_t))
    return x, u

def plot_paths(x1,x2,ylabel,R1,R2):
    fig, ax = plt.subplots()
    colors = [ '#2D328F', '#F15C19' ] # blue, orange
    label_fontsize = 18

    t = np.arange(0,x1.shape[0])
    ax.plot(t,x1,color=colors[0],label='R={}'.format(R1))
    ax.plot(t,x2,color=colors[1],label='R={}'.format(R2))
        
    ax.set_xlabel('time',fontsize=label_fontsize)
    ax.set_ylabel(ylabel,fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return
    
A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R1 = 0
R2 = 10

x0 = np.array(-1).reshape(1,1)

# number of time steps to simulate
T = 50

K_1 = dlqr(A,B,Q,R1)
(x_1,u_1) = simulate(A,B,K_1,x0,T)

K_2 = dlqr(A,B,Q,R2)
(x_2,u_2) = simulate(A,B,K_2,x0,T)

    
plot_paths(x_1[0],x_2[0],'Position',R1,R2)
plot_paths(u_1.T,u_2.T,'Control Action',R1,R2)
