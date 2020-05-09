import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R,gamma=1):
    '''
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    '''

    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, B, Q, R/gamma)

    F = gamma*np.matmul(scipy.linalg.inv(np.matmul(np.matmul(B.T, gamma*P), B) + R), (np.matmul(np.matmul(B.T, P), A)))
    return -F, P

# %% 1-D

A = np.array([[5]])
B = np.array([[1]])
Q = np.array([[1]])
R = np.array([[5]])

K = np.array([[5]])
K_0 = np.copy(K)

print(f"K = {K}")
gamma = min(1/((A+np.matmul(B,K_0))**2).item(), 1)
print(f"gamma = {gamma}")

while np.abs(A+np.matmul(B,K_0)).item() >= 1:
    K, P = dlqr(A,B,Q,R,gamma)
    print(f"K' = {K}")
    
    print("optimal to initial:"
        f"{np.abs(A+np.matmul(B,K)).item()/np.abs(A+np.matmul(B,K_0)).item()}")
    
    print("estimate bound:"
          f"{A.item() * np.sqrt(R.item()) / (2 * B.item() * np.sqrt(P.item()))}")
    
    print("\n")
    
    K_0 = np.copy(K)
    
    gamma = min(1/((A+np.matmul(B,K_0))**2).item(),1)
    print(f"gamma = {gamma}")
    
   
    
K_star, _ = dlqr(A,B,Q,R)
print(f"K* = {K_star}")
        

x = np.linspace(0.01,1,100)
y = np.array([A.item()*np.sqrt(R.item())/2/B.item()/np.sqrt(np.abs(scipy.linalg.solve_discrete_are(np.sqrt(gamma)*A, B, Q, R/gamma))).item() 
    for gamma in x]).squeeze()

fig, ax = plt.subplots()

ax.plot(x,y,'k-')

ax.set_xlabel('gamma')
ax.set_ylabel('P')

plt.show()
