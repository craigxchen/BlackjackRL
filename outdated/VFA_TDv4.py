from VFA_Net import NeuralNetwork
from collections import defaultdict
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gym
import random
import pickle
import numpy as np

nn_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "tanh"},
]

def MSE_Loss(x, y):
    return 0.5*(x-y)**2

batch_size = 32
alpha = 1
gamma = 1
epsilon = 1
num_trials = 200000

model = NeuralNetwork(nn_arq, batch_size)
env = gym.make('Blackjack-v0')

N = defaultdict(lambda: 0) # N table
# %% 

def process(state):
    return np.array([state[0], state[1], state[2]]).reshape((3,1))

def policy(state, epsilon=epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        next_sums = [k + state[0] for k in list(range(1,11)) + 3*[10]]
        for k in range(len(next_sums)):
            if state[2] and next_sums[k] > 21:
                next_sums[k] -= 10

        next_states = [(x, state[1], state[2]) for x in next_sums]
        next_values = [model(process(s)) for s in next_states]
        
        EV_hit = np.mean(next_values)
        EV_stay = model(process(state))
        action = EV_hit > EV_stay
    return action[0][0]

def get_policy():
    P = {}
    for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] 
        + [(x, y, False) for x in range(4,22) for y in range(1, 11)]):
        P[k] = policy(k, 0)
    return P

def plot_policy(policy, usable_ace = False):
    if not usable_ace:
        data = np.empty((18, 10))
        for state in policy.keys():
            if state[2] == usable_ace: 
                data[state[0]-4][state[1]-1] = policy[state]
    else:
        data = np.empty((10, 10))
        for state in policy.keys():
            if state[2] == usable_ace:
                data[state[0]-12][state[1]-1] = policy[state]
    
    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'green'])
    bounds = [-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    
    ax.set_xticks(np.arange(10))
    if not usable_ace:
        ax.set_yticks(np.arange(18))
        ax.set_yticklabels(['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])
    else:
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(['A + 1', 'A + 2', 'A + 3', 'A + 4', 'A + 5', 'A + 6', 'A + 7', 'A + 8', 'A + 9', 'A + 10'])
        
    ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    
    ax.set_xlabel('Dealer Show Card')
    if not usable_ace:
        ax.set_ylabel('Player Sum (Hard)')
    else:
        ax.set_ylabel('Player Sum (Soft)')
    
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.invert_yaxis()
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.4)
    # manually define a new patch 
    patch1 = mpatches.Patch(color='green', label='Hit')
    patch2 = mpatches.Patch(color='red', label='Stand')   
    # plot the legend
    plt.legend(handles=[patch1, patch2], loc='upper right')
    
    plt.show()
    return
        
# %% training
with open("near_optimal", 'rb') as f:
    P_star = pickle.load(f)    

for i in range(num_trials):
    if (i+1)%(num_trials/10) == 0:
        print('trial {}/{}'.format(i+1,num_trials))
        
    epsilon = max(epsilon*0.99995, 0.001)
        
    state = env.reset()
    done = False
    
    while not done:
        N[state]+=1
        action = P_star[state]
        
        batch = []
        for i in range(batch_size):
            next_state, reward, done, _ = env.step(action)
            if state == next_state:
                y = reward
            else:
                y = reward + gamma*alpha*model(process(next_state))
            batch.append(y)
        
        y = np.mean(batch)
        
        y_hat = alpha*model.net_forward(process(state))
        
        lr = 1/(alpha*(1+N[state])**0.85)
        
        model.net_backward(y_hat, y)
        model.update_wb(lr)
        
        state = next_state          

P_derived = get_policy()
V = dict((k,model(process(k))) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
plot_policy(P_derived)
plot_policy(P_derived, True)


