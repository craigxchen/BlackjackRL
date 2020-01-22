import sys, os
# access to files one directory up
sys.path.append(os.path.abspath(os.path.join("..")))

import numpy as np
import matplotlib.pyplot as plt
from PG_Net import PGNet
from VFA_Net import NeuralNetwork
from blackjack_complete import Blackjack
import blackjack_plot_tools as bpt

act_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 2, "activation": "softmax"},
]

ctc_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "none"},
]

ACTOR = PGNet(act_arq, bias=True, double=True)
CRITIC = NeuralNetwork(ctc_arq, bias=True, double=True)

ALPHA = 100
GAMMA = 1
NUM_TRIALS = 500000
BATCH_SIZE = 1

def loss(target, prediction, alpha=1):
    return float((target-alpha*prediction)**2)

env = Blackjack()

# %% 

def process(state):
    """
    inputs a state, returns a parameterization of the state
    
    uncomment return statements for different types of feature mappings, 
    don't forget to change the input dim of the neural net
    """
    
    ## normalized vector
    return np.array([state[0]/(max(env.state_space)[0]), state[1]/(max(env.state_space)[1]), state[2]]).reshape((3,1))

def plot_loss(y):
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(t[10::int(NUM_TRIALS/500)],y[10::int(NUM_TRIALS/500)])
        
    ax.set_yscale('log')
    ax.set_xlabel('Trials',fontsize=label_fontsize)
    ax.set_ylabel('Loss',fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

# %% training
def train(**kwargs):
    loss_history = []
    for i in range(NUM_TRIALS):
        if (i+1)%(NUM_TRIALS/10) == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS))
            
        state = env.reset()
        done = False
        
        A_grads = []
        C_grads = []
        
        for j in range(BATCH_SIZE):
            while not done:
                probs = ACTOR.net_forward(process(state))
                action = np.random.choice(env.action_space, p=probs.flatten())
                next_state, reward, done = env.step(action)
    
                if action == 1:
                    ## compute expected value
                    fut_vals = []
                    fut_states, _ = env.future_states(state)
                    for fs in fut_states:
                        if fs[0] > 21: #bust = terminal state, so v = 0
                            fut_vals.append(np.array(reward).reshape(1,1))
                        else:
                            fut_vals.append(reward + ALPHA*GAMMA*CRITIC(process(fs)))
                    
                    y = np.mean(fut_vals)
                    
                    ## sampling
#                    y = reward + ALPHA*GAMMA*CRITIC(process(next_state))
                else:
                    y = np.array(reward).reshape(1,1)
                
                y_hat = CRITIC.net_forward(process(state))
                
                loss_history.append(loss(y, y_hat, ALPHA))
                
                A_grads.append(ACTOR.net_backward(y-y_hat, probs))
                C_grads.append(CRITIC.net_backward(y, y_hat, ALPHA))
                
                state = next_state
    
        lr = 0.001
        if (i+1)%100 == 0:
            ACTOR.batch_update_wb(lr, A_grads)
        CRITIC.batch_update_wb(lr, C_grads)
        
    
    V = dict((k,ALPHA*CRITIC(process(k)).item()) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    P = dict((k,np.argmax(ACTOR(process(k)))) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    
    return P, V, loss_history

#model.reset_params()
P, V, loss_history = train()
plot_loss(loss_history)
bpt.plot_policy(P, False)
bpt.plot_policy(P, True)
bpt.plot_v(V)
bpt.plot_v(V, True)

