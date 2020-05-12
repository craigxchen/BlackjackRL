import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R,gamma=1):
    '''
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    '''

    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, B, Q, R/gamma)

    F = -gamma*np.matmul(scipy.linalg.inv(gamma*np.matmul(np.matmul(B.T, P), B) + R), (np.matmul(np.matmul(B.T, P), A)))
    return F, P

DIM = 3
#SEED = 9
#np.random.seed(SEED)

A = np.random.randn(DIM,DIM)
B = np.random.randn(DIM,DIM)
B = np.identity(DIM)
Q = np.identity(DIM)
R = np.identity(DIM)


K = np.random.randn(DIM,DIM)
K_0 = np.copy(K)

print(f"K = {K}")
gamma = min(1/max(np.linalg.eig(A+np.matmul(B,K))[0].real)**2,1)
print(f"gamma = {gamma}")

print("\n")

while max(np.linalg.eig(A+np.matmul(B,K))[0].real)**2 >= 1:
    K, P = dlqr(A,B,Q,R,gamma)
    print(f"K' = {K}")
    
    print(f"optimal to initial: {np.abs(max(np.linalg.eig(A+np.matmul(B,K))[0].real)/max(np.linalg.eig(A+np.matmul(B,K_0))[0].real))}")
    
    print("\n")
    
    gamma = min(1/(max(np.linalg.eig(A+np.matmul(B,K))[0].real))**2,1)
    print(f"gamma = {gamma}")
    K_0 = np.copy(K)
    
K_star, _ = dlqr(A,B,Q,R)
print(f"K* = {K_star}")
        

x = np.linspace(0.01,1,100)

w = np.array([1/np.sqrt(gamma) for gamma in x]).squeeze()
v = np.array([np.abs(max(np.linalg.eig(A+np.matmul(B,dlqr(A,B,Q,R,gamma)[0]))[0].real)) for gamma in x]).squeeze()

fig, ax = plt.subplots()

ax.plot(x,w,'r-', label="initial")
ax.plot(x,v,'b-', label="new optimal")

ax.set_xlabel('gamma')
ax.set_ylabel("spectral radius")

plt.legend()
plt.show()