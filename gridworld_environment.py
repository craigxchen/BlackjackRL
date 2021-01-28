import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator


class GridWorld:
    def __init__(self, n=50, random_init=False):
        self.n = n
        
        if random_init:
            self.reset = self.reset_rand
        else:
            self.reset = self.reset_static
        
        self.state_space = [[a, b] for a in range(self.n + 1) for b in range(3)]
        self.action_space = [0, 1]  # move down, move right
        self.obs_dim = 2
        self.action_dim = len(self.action_space)
        
        self.start = [0, 0]
        self.final = [[a, 2] for a in range(self.n)] + [[self.n, b] for b in range(3) if b != 1]
        
        # tracking agent info        
        self.curr_state = self.start
        
    def step(self, action):
        assert action in self.action_space
        x, y = self.curr_state
        delta = [0, 1] if action else [1, 0]
        
        if self.curr_state == [self.n, 1]:
            self.curr_state = [0,1]
        else:
            self.curr_state = [x + delta[0], y + delta[1]]
        
        # return obs, reward, done
        return np.array([self.curr_state]), self.get_reward(self.curr_state), self.curr_state in self.final
    
    def reset_static(self):
        self.curr_state = self.start
        # return observation, reward, done
        return self.start, 0.0, False
    
    def reset_rand(self):
        x = np.random.randint(0, self.n)
        self.curr_state = [x, 0]
        return self.curr_state, 0.0, False


    def get_reward(self, state):
        x, y = self.curr_state
        if y == 1:
            return -(2*self.n)
        elif self.curr_state in self.final:
            return 0
        else:  # y == 0
            return -1
        
    def plot_policy(self, model):
        data = np.empty((51, 3))
        for state in self.state_space:
            data[state[0],state[1]] = np.argmax(model(torch.tensor(state).float()).detach().numpy())
        
        # create discrete colormaps
        cmap = colors.ListedColormap(['green', 'red'])
        bounds = [-0.5,0.5,1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, norm=norm)
        
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(51))
        
        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.invert_yaxis()
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.4)
        # manually define a new patch 
        patch1 = mpatches.Patch(color='green', label='Down')
        patch2 = mpatches.Patch(color='red', label='Right')   
        # plot the legend
        plt.legend(handles=[patch1, patch2], loc='upper right')
        
        plt.show()
        return

    
    
    
    