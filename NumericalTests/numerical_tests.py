import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def dlqr(A, B, Q, R, gamma=1):
    """
    Solves for the optimal infinite-horizon, discrete-time LQR controller
    given linear system (A,B) and cost function parameterized by (Q,R)
    """

    P = scipy.linalg.solve_discrete_are(np.sqrt(gamma) * A, B, Q, R / gamma)

    F = gamma * np.matmul(scipy.linalg.inv(np.matmul(np.matmul(B.T, gamma * P), B) + R),
                          (np.matmul(np.matmul(B.T, P), A)))
    return -F, P


# %% 1-D

A = np.array([[10]])
B = np.array([[1]])
Q = np.array([[1]])
R = np.array([[1]])

K = np.array([[5]])
K_0 = np.copy(K)

print(f"K = {K}")
gamma = min(1 / ((A + np.matmul(B, K_0)) ** 2).item(), 1)
print(f"gamma = {gamma}")

print("\n")

while np.abs(A + np.matmul(B, K_0)).item() >= 1:
    K, P = dlqr(A, B, Q, R, gamma)
    print(f"K' = {K}")

    print("optimal to initial:"
          f"{np.abs(A + np.matmul(B, K)).item() / np.abs(A + np.matmul(B, K_0)).item()}")

    print("estimate bound:"
          f"{A.item() * np.sqrt(R.item()) / (2 * B.item() * np.sqrt(P.item()))}")

    print("\n")

    K_0 = np.copy(K)

    gamma = min(1 / ((A + np.matmul(B, K_0)) ** 2).item(), 1)
    print(f"gamma = {gamma}")

K_star, _ = dlqr(A, B, Q, R)
print(f"K* = {K_star}")

x = np.linspace(0.01, 1, 100)
y = np.array([1 / np.sqrt(gamma) * A.item() * np.sqrt(R.item()) / 2 / B.item() / np.sqrt(
    np.abs(scipy.linalg.solve_discrete_are(np.sqrt(gamma) * A, B, Q, R / gamma))).item()
              for gamma in x]).squeeze()
w = np.array([1 / np.sqrt(gamma) for gamma in x]).squeeze()
v = np.array([np.abs(A + np.matmul(B, dlqr(A, B, Q, R, gamma)[0])).item() for gamma in x]).squeeze()

fig, ax = plt.subplots()

ax.plot(x, y, 'k-', label="upper bound on new opt")
ax.plot(x, w, 'r-', label="initial")
ax.plot(x, v, 'b-', label="new optimal")

ax.set_xlabel('gamma')
ax.set_ylabel('P')

plt.legend()
plt.show()

# %% bound constant as function of A,R

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 20, 0.1)
Y = np.arange(0, 20, 0.1)
X_grid, Y_grid = np.meshgrid(X, Y)

Z = np.array([[x * np.sqrt(y) / np.sqrt(dlqr(x.reshape(1, 1), B, Q, y.reshape(1, 1))[1]).item() for x in X] for y in Y])

# Plot the surface.
surf = ax.plot_surface(X_grid, Y_grid, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel("A")
ax.set_ylabel("R")
ax.set_zlabel("A*sqrt(R)/sqrt(P)")

plt.show()
