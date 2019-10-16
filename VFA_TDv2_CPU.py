from blackjack_complete_TEST import CompleteBlackjackEnv
import random
import numpy as np
from collections import defaultdict
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F

# N is Batch Size; D_in is input dimension; 
# H is hidden dimension; D_out is output dimension
N, D_in, D_H, D_out = 1, 3, 512, 1

num_trials = 500000
alpha = 1 # scaling parameter
#lr = 1e-3 # learning rate
gamma = 1.0 # discount rate
epsilon = 1.0 # exploration rate

# %%
class OneHiddenLayer(nn.Module):
    def __init__ (self, D_in, D_H, D_out):
        super().__init__()
        self.layer1 = nn.Linear(D_in, D_H)
        tempw1 = torch.randn(int(D_H/2), D_in) * (1/np.sqrt(D_H))
        self.layer1.weight = torch.nn.Parameter(torch.cat((tempw1, -tempw1))) # doubling trick
        tempb1 = torch.randn(int(D_H/2)) * (1/np.sqrt(D_H))
        self.layer1.bias = torch.nn.Parameter(torch.cat((tempb1, -tempb1))) # doubling trick
        
        self.layer2 = nn.Linear(D_H, D_out, bias=False)
        tempw2 = torch.randn(D_out, int(D_H/2)) * (1/np.sqrt(D_out))
        self.layer2.weight = torch.nn.Parameter(torch.cat((tempw2, -tempw2), 1)) # doubling trick

        # no doubling for the bias as it is a 1x1 matrix

    def forward(self, a0):
        return self.layer2(F.relu(self.layer1(a0)))
    
def preprocess(state):
    return torch.tensor([float(state[0]), float(state[1]), float(state[2])])

def unprocess(state):
    return (int(state.detach().numpy()[0]), int(state.detach().numpy()[1]), int(state.detach().numpy()[2]))

def policy(state):
    if random.uniform(0,1) < epsilon:
        action = random.choice(env.action_space)
    else:
        next_states = [torch.tensor([float(hands), float(state[1]), float(state[2])]) for hands in env.future_hands()]
        next_values = []
        for s in next_states:
            if s[0] > 21:
                next_values.append(0.0)
            else: 
                next_values.append(VFA(s).cpu().detach().numpy())
        
        EV_hit = np.mean(next_values)
        EV_stay = float(VFA(state).cpu().detach().numpy())
        action = EV_hit > EV_stay
    return action

def get_policy(state):
    next_states = [torch.tensor([float(hands), float(state[1]), float(state[2])]) for hands in env.future_hands()]
    EV_hit = np.mean([VFA(s).cpu().detach().numpy() for s in next_states])
    EV_stay = float(VFA(state).cpu().detach().numpy())
    action = EV_hit > EV_stay
    return action

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
#    patch3 = mpatches.Patch(color='blue', label='Double Down')    
    # plot the legend
    plt.legend(handles=[patch1, patch2], loc='upper right')
    
    plt.show()
    return  

def save_policyplot():
    
    return

# %% training
VFA = OneHiddenLayer(D_in, D_H, D_out)

env = CompleteBlackjackEnv()

loss_fn = nn.MSELoss()

N = defaultdict(lambda: 0)
for idx in range(num_trials):
    if (idx+1)%(num_trials/10) == 0:
        print("trial {}/{}".format(idx + 1, num_trials))
    state, reward = env.reset()
    done = False
    N[state] += 1
    state = preprocess(state)
    while not done:
        action = policy(state)
        next_state, next_reward, done = env.step(action)
        
        y = reward + alpha*gamma*VFA(preprocess(next_state)) # TD(0) Target
        y_hat = alpha*VFA(state)

        loss = loss_fn(y_hat, y)

        VFA.zero_grad()
        loss.backward()
        
        lr = 1/(1 + N[unprocess(state)])**0.85
        with torch.no_grad():
            for p in VFA.parameters(): 
                p -= (lr/(2*alpha)) * p.grad
        
        state = preprocess(next_state)
        reward = next_reward
        N[unprocess(state)] += 1
        
    # run one more time to update the value of the terminal state
    y = torch.tensor([float(reward)])
    y_hat = alpha*VFA(state)

    loss = loss_fn(y_hat, y)

    VFA.zero_grad()
    loss.backward()
    
    lr = 1/(1 + N[unprocess(state)])**0.85
    with torch.no_grad():
        for p in VFA.parameters(): 
            p -= (lr/(2*alpha)) * p.grad
    
    if idx < 0.3*num_trials:
        epsilon *= 0.999995
    elif 0.3*num_trials <= idx < 0.8*num_trials:
        epsilon *= 0.99995
    else:        
        epsilon = max(epsilon*999995, 0.005)

V = dict((k,v.cpu().detach().numpy()) for k in env.state_space for v in VFA(preprocess(k)))
P = dict((k,get_policy(preprocess(k))) for k in env.state_space)
plot_policy(P)
#save_policyplot()
