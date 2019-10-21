from VFA_Net import NeuralNetwork
from collections import defaultdict
import gym
import random
import numpy as np

nn_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "none"},
]

def MSE_Loss(x, y):
    return 0.5*(x-y)**2

batch_size = 1
alpha = 100
gamma = 1
epsilon = 1
num_trials = 50

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
        
# %% training
for i in range(num_trials):
        if (i+1)%(num_trials/10) == 0:
            print('trial {}/{}'.format(i+1,num_trials))
            
        epsilon = max(epsilon*0.99995, 0.001)
            
        state = env.reset()
        done = False
        
        while not done:
            N[state]+=1
            # if policy is a dictionary
            if isinstance(policy, dict):
                action = policy[state]
            else: # policy is a function
                action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            y_hat = alpha*model.net_forward(process(state))
            y = reward + gamma*alpha*model(process(next_state))
            
            lr = 1/(alpha*(1+N[state])**0.85)
            
            model.net_backward(y_hat, y)
            model.update_wb(lr)
            
            state = next_state          

P = dict((k,policy(process(k), 0)) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
V = dict((k,model(process(k))) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
