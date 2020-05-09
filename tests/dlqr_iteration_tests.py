import numpy as np
import scipy.linalg
import scipy.sparse.linalg as sp
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R,gamma=1):
    '''
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    '''

    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, B, Q, R/gamma)

    F = gamma*np.matmul(scipy.linalg.inv(np.matmul(np.matmul(B.T, gamma*P), B) + R), (np.matmul(np.matmul(B.T, P), A)))
    return -F, P

DIM = 3
#SEED = 9
#np.random.seed(SEED)

A = np.random.randn(DIM,DIM)
#B = np.random.randn(DIM,DIM)
B = np.identity(DIM)
Q = np.identity(DIM)
R = np.identity(DIM)

#K = np.array([[0.,0.,1.],[0.,0.5,1.],[0.,0.,1.]])
K = np.random.randn(DIM,DIM)
K_0 = np.copy(K)

print(f"K = {K}")
gamma = min(1/(sp.svds(A+np.matmul(B,K),1)[1]**2),1)
print(f"gamma = {gamma}")

while sp.svds(A+np.matmul(B,K),1)[1]**2 >= 1:
    K, P = dlqr(A,B,Q,R,gamma)
    print(f"K' = {K}")
    
    print(f"optimal to initial: {sp.svds(A+np.matmul(B,K),1)[1]/sp.svds(A+np.matmul(B,K_0),1)[1]}")
    
    gamma = min(1/(sp.svds(A+np.matmul(B,K),1)[1]**2),1)
    print(f"gamma = {gamma}")
    K_0 = np.copy(K)
    
K_star, _ = dlqr(A,B,Q,R)
print(f"K* = {K_star}")
        

x = np.linspace(0.01,1,100)
y = np.array([sp.svds(scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, B, Q, R/gamma),1)[1] for gamma in x]).squeeze()

fig, ax = plt.subplots()

ax.plot(x,y,'k-')

ax.set_xlabel('gamma')
ax.set_ylabel("max singular value of P")

plt.show()