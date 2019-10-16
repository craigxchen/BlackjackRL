from blackjack_complete_TEST import CompleteBlackjackEnv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor

class VFA(nn.Module):
    def __init__ (self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim) / np.sqrt(hidden_dim)) # change
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.w2 = nn.Parameter(torch.randn(output_dim, hidden_dim) / np.sqrt(output_dim)) # change
        self.b2 = nn.Parameter(torch.zeros(output_dim, 1))

    def forward(self, x0):
        z1 = x0.mm(self.w1) + self.b1
        a1 = F.relu(z1)
        z2 = a1.mm(self.w2) + self.b2
        return z2

def preprocess(state):
    return torch.tensor([float(state[0]), float(state[1]), float(state[2])])

def TD_error(state, reward, next_state):
    return (1/alpha) * (reward + gamma*alpha*model(next_state) - alpha*model(state))

def update_w(td_error):
    with torch.no_grad():
        model.zero_grad()
        for p in model.parameters(): p -= p.grad * lr * -td_error # derivative of TD error is (-td_error) * dV/dTheta where Theta is a parameter
    return

def policy(state):
    #epsilon-greedy
    if random.uniform(0, 1) <= epsilon:
        action = random.choice(env.action_space)
    else:
        action = 0
        # FIX
        # Compute transition probabilities
    return action

def learn(state, action, reward, next_state):
    td_error = TD_error(state, reward, next_state)
    update_w(td_error)
    return

def play(policy):
        # start
        state = preprocess(env.reset())
        history = []
        done = False
        
        while not done:
            # if policy is a dictionary
            if isinstance(policy, dict):
                action = policy[state]
            else: # policy is a function
                action = policy(state)
            next_state, reward, done = env.step(action)
            history.append([state, action, reward, done])
            state = next_state

        return history
    
model = VFA(3, 512, 1)

env = CompleteBlackjackEnv()

gamma = 1
alpha = 100
epsilon = 1
lr = 1e-3
num_trials = 1000

#for idx in range(num_trials):
    # add here
    