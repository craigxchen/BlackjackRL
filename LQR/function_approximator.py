import numpy as np
import matplotlib.pyplot as plt

class LQSystem:
    def __init__(self, A, B, Q, R):
        """A class used to group together the linear-quadratic system described
        by matrices A, B, Q, and R. 
        
        Specifically: 
            x_t+1 = A*x_t + B*u_t
            c_t+1 = x_t.T*Q*x_t + u_t.T*R*u_t
        
        ...
        
        Attributes
        ----------

        Methods
        -------

        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
    def __call__(self, x0, u0):
        return self.step(x0, u0), self.cost(x0, u0)
        
    def cost(self, x0, u0):
        """Calculates cost for given state and action
        
        c_t+1 = x_t.T*Q*x_t + u_t.T*R*u_t
        
        Parameters
        ----------
        x0 : column vector
            Numpy Array containing parameterization of current state
            
        u0 : column vector
            Numpy Array containing chosen action
            
        Returns
        -------
        c_t+1 : float (in 1x1 array)
            Numpy Array containing cost of associated state and action
        """
        return np.matmul(np.matmul(self.Q, x0), x0) + np.matmul(np.matmul(self.R, u0), u0)
    
    def step(self, x0, u0):
        """Yields next state
        
        x_t+1 = A*x_t + B*u_t
        
        Parameters
        ----------
        x0 : column vector
            Numpy Array containing parameterization of current state
            
        u0 : column vector
            Numpy Array containing chosen action
            
        Returns
        -------
        x_t+1 : column vector
            Numpy Array containing parameterization of next state
        """
        return np.matmul(self.A, x0) + np.matmul(self.B, u0)

class Approximator:
    def __init__(self, structure):
        self.structure = structure
        
        