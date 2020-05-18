import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def dlqr(A, B, Q, R, gamma=1):
    """
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    """

    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma) * A, B, Q, R / gamma)

    F = -gamma * np.matmul(scipy.linalg.inv(gamma * np.matmul(np.matmul(B.T, P), B) + R),
                           (np.matmul(np.matmul(B.T, P), A)))
    return F, P


DIM = 10
#SEED = 11
#np.random.seed(SEED)

A = np.random.randn(DIM, DIM)
B = np.random.randn(DIM, DIM)

#B = np.identity(DIM)
Q = np.identity(DIM)
R = np.identity(DIM)

K = np.random.randn(DIM, DIM)
K_0 = np.copy(K)

#print(f"K = {K}")
gamma = min(1 / max(np.abs(np.linalg.eig(A + B@K)[0])) ** 2, 1)
gamma_0 = gamma

print(f"gamma = {gamma}")

print("\n")

K_star, P_star = dlqr(A, B, Q, R)

print(f"iter bound: {2*np.trace(P_star)/min(np.linalg.eig(Q)[0]) * np.log(1/np.sqrt(gamma))}")
print(f"iter bound: {2*max(np.linalg.eig(P_star)[0])/min(np.linalg.eig(Q)[0]) * np.log(1/np.sqrt(gamma))}")

print("\n")

i=0
while max(np.abs(np.linalg.eig(A + B@K)[0])) ** 2 >= 1: 
    i += 1
    K, P = dlqr(A, B, Q, R, gamma)
#    print(f"K' = {K}")

    print("optimal to initial:"
          f"{np.abs(max(np.linalg.eig(A + B@K)[0])) / np.abs(max(np.linalg.eig(A + B@K_0)[0]))}")

    print("estimate bound:"
          f"{np.sqrt(1 - min(np.linalg.eig(Q + K.T@R@K)[0]) / max(np.linalg.eig(P)[0]))}")

    print("uniform bound:"
          f"{np.sqrt(1-(min(np.linalg.eig(Q)[0])/np.trace(P_star)))}")

    print("\n")

    gamma = min(1 / max(np.abs(np.linalg.eig(A + B@K)[0])) ** 2, 1)
    print(f"gamma = {gamma}")
    K_0 = np.copy(K)


#print(f"K* = {K_star}")

print(f"real num iter: {i}")

# %% improvement ratio v bound plot

x = np.linspace(0.01, 1, 500)

true_ratios = np.array(
    [np.sqrt(gamma) * max(np.abs(np.linalg.eig(A + B @ dlqr(A, B, Q, R, gamma)[0])[0])) for gamma in x]).squeeze()
temp = []
for gamma in x:
    k, p = dlqr(A, B, Q, R, gamma)
    temp.append(np.sqrt(1 - min(np.linalg.eig(Q + k.T@R@k)[0]) / max(np.linalg.eig(p)[0])))
    
est_ratios = np.array(temp).squeeze()

ub_ratios = np.array([np.sqrt(
        1-(min(np.linalg.eig(Q)[0])/np.trace(P_star))) for _ in x]).squeeze()

fig1, ax1 = plt.subplots()

ax1.plot(x, true_ratios, 'b-', label="True Ratio")
ax1.plot(x, est_ratios, 'k-', label="Estimate (gamma-dependent)")
ax1.plot(x, ub_ratios, 'r-', label="Uniform Bound")

ax1.set_xlabel("gamma")
ax1.set_ylabel(r'$\rho (A+BK^*) / \rho (A+BK_0)$')

fig1.legend(loc="center right", bbox_to_anchor=(0.8,0.5))
fig1.show()

# %% iterations v bound plot

fig2, ax2 = plt.subplots()

tmp = []
for gamma in x:
    i = 0
    while gamma < 1:
        i += 1
        K, P = dlqr(A, B, Q, R, gamma)
        K_0 = np.copy(K)
    
        gamma = min(1 / max(np.abs(np.linalg.eig(A + B@K)[0])) ** 2, 1)
        
    tmp.append(i)
    
tr = np.array(tmp).squeeze()

est = np.array([2*np.trace(dlqr(A,B,Q,R,gamma)[1])/min(np.linalg.eig(Q)[0]) * np.log(1/np.sqrt(gamma)) for gamma in x]).squeeze()
ub = np.array([2*np.trace(dlqr(A,B,Q,R)[1])/min(np.linalg.eig(Q)[0]) * np.log(1/np.sqrt(gamma)) for gamma in x]).squeeze()

ax2.plot(x, tr, 'b-', label="True # of Iterations")
#ax2.plot(x, est, 'k-', label="Estimate (gamma-dependent)")
ax2.plot(x, ub, 'r-', label="Theorem 3 Bound")

ax2.set_yscale('log')

ax2.set_xlim(0,1)
#ax2.set_ylim(bottom=0)

ax2.set_xlabel('gamma')
ax2.set_ylabel('iterations (log-scale)')

fig2.legend(loc="upper right", bbox_to_anchor=(0.9,0.85))
fig2.show()
