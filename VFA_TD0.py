from VFA_Net import NeuralNetwork
from blackjack_complete import Blackjack
import blackjack_plot_tools as bpt
import matplotlib.pyplot as plt
import pickle
import numpy as np

nn_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "none"},
]

ALPHA = 100
GAMMA = 1
NUM_TRIALS = 100000
BATCH_SIZE = 1

def loss(target, prediction, alpha=1):
    return float((target-alpha*prediction)**2)

model = NeuralNetwork(nn_arq, bias=True, double = True, seed=1, initVar = 1, initVarLast = 0)
   
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
    
    ## one-hot (dim = 280,1)
#    return np.array([int(state == k) for k in env.state_space]).reshape(len(env.state_space),1)

def get_policy(V):
    P = {}
    for state in V.keys():
        EV_Hit = []
        EV_Stay = model(process(state))
        
        fut_states, _ = env.future_states(state)
        for fs in fut_states:
            if fs[0] <= 21: 
                EV_Hit.append(model(process(fs)))
            else: #sets terminal states to zero
                EV_Hit.append(np.array(0).reshape(1,1))
        
        if not EV_Hit:
            EV_Hit.append(-1)

        P[state] = (np.mean(EV_Hit) > EV_Stay)[0][0].astype(int)
    return P

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
with open("input_policy", 'rb') as f:
    P_star = pickle.load(f)    

def train(**kwargs):
    loss_history = []
    for i in range(NUM_TRIALS):
        if (i+1)%(NUM_TRIALS/10) == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS))
            
        state = env.reset()
        done = False
        
        grad_values = []
        
        for j in range(BATCH_SIZE):
            while not done:
                action = P_star[state]
                next_state, reward, done = env.step(action)
    
                if action == 1:
                    ## compute expected value
                    fut_vals = []
                    fut_states, _ = env.future_states(state)
                    for fs in fut_states:
                        if fs[0] > 21: #bust = terminal state, so v = 0
                            fut_vals.append(np.array(reward).reshape(1,1))
                        else:
                            fut_vals.append(reward + ALPHA*GAMMA*model(process(fs)))
                    
                    y = np.mean(fut_vals)
                    
                    ## sampling
#                    y = reward + ALPHA*GAMMA*model(process(next_state))
                else:
                    y = np.array(reward).reshape(1,1)
                
                y_hat = model.net_forward(process(state))
                
                loss_history.append(loss(y, y_hat, ALPHA))
                grad_values.append(model.net_backward(y, y_hat, ALPHA))
                
                state = next_state
    
        lr = 0.001
        model.batch_update_wb(lr, grad_values)
        
    
    V = dict((k,ALPHA*model(process(k)).item()) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    P_derived = get_policy(V)
    
    return P_derived, V, loss_history

#model.reset_params()
P_derived, V, loss_history = train()
plot_loss(loss_history)
bpt.plot_policy(P_derived, False)
bpt.plot_policy(P_derived, True)
bpt.plot_v(V)
bpt.plot_v(V, True)
