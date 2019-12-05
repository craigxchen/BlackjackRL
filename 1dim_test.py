import numpy as np
from lqr_control import control 

A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R1 = np.array(1).reshape(1,1)
R2 = np.array(10).reshape(1,1)

x0 = np.array(-1).reshape(1,1)
u0 = np.array(0).reshape(1,1)

# number of time steps to simulate
T = 30

K_1, _, _ = control.dlqr(A,B,Q,R1)
x_1, u_1 = control.simulate_discrete(A,B,K_1,x0,u0,T)

K_2, _, _ = control.dlqr(A,B,Q,R2)
x_2, u_2 = control.simulate_discrete(A,B,K_2,x0,u0,T)

    
control.plot_paths(x_1[0],x_2[0],'Position',R1,R2)
control.plot_paths(u_1.T,u_2.T,'Control Action',R1,R2)
