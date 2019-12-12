import numpy as np
import lqr_control as control 

A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
B = np.identity(3)
Q = np.identity(3)
R1 = np.identity(3)*100
R2 = np.identity(3)*1000

x0 = np.array([[1],[0],[-1]]) #(0,0,0) is stable
u0 = np.zeros((3,1)) # default to 0 init for starting action

# number of time steps to simulate
T = 100

K_1, _, _ = control.dlqr(A,B,Q,R1)

K_2, _, _ = control.dlqr(A,B,Q,R2)

x_1, u_1 = control.simulate_discrete(A,B,K_1,x0,u0,T)
x_2, u_2 = control.simulate_discrete(A,B,K_2,x0,u0,T)

control.plot_states(x_1, 'State Temps', R1)
control.plot_states(x_2, 'State Temps', R2)