import numpy as np


class GridWorld:
    def __init__(self, n=50):
        self.n = n

        self.action_space = [0, 1]  # move down, move right
        self.obs_dim = 1
        self.action_dim = len(self.action_space)
        
        self.start = 0
        self.final = 0
        
        # tracking agent info        
        self.curr_state = self.start
        
    def step(self, action):
        assert action in self.action_space
        if action == 0:  # move down
            if self.curr_state == self.n:
                self.curr_state = -1
            elif self.curr_state == 2*self.n:
                self.curr_state = -1
            else:
                self.curr_state += 1
        else:  # move right
            if self.curr_state == 2*self.n:
                self.curr_state = self.curr_state
            else:
                self.curr_state = 2*self.n
        
        # return obs, reward, done
        return np.array([self.curr_state, self.get_reward(self.curr_state), self.curr_state == -1])
    
    def reset(self):
        self.curr_state = self.start
        # return observation, reward, done
        return self.start, 0.0, False


    def get_reward(self, state):
        if state < 0:
            return 0
        elif 0 < state <= self.n:
            return -1
        else:
            return -(2*self.n)
    
    
    
    