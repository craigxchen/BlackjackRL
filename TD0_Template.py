from VFA_Net import NeuralNetwork
import matplotlib.pyplot as plt
import pickle
import numpy as np

nn_arq = [
    # example for formatting
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "none"},
]

ALPHA = 100
GAMMA = 1
NUM_TRIALS = 100000
BATCH_SIZE = 1

def loss(target, prediction, alpha=1):
    return float((target-alpha*prediction)**2)

MODEL = NeuralNetwork(nn_arq, bias=True, double=False, zero=True, seed=1)
   
ENV = "CHANGEME" # your environment here - be sure it has a state space

# %% 

def process(state):
    """
    inputs a state, returns a parameterization of the state
    """
    return np.random((1,1)) # must be column vector (2D) that corresponds with input dim

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
    
#with open("input_policy", 'rb') as f:
#    P_star = pickle.load(f)    

P_star = "CHANGEME"

def train(**kwargs):
    loss_history = []
    for i in range(NUM_TRIALS):
        if (i+1)%(NUM_TRIALS/10) == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS))
            
        state = ENV.reset()
        done = False
        
        grad_values = []
        
        for j in range(BATCH_SIZE):
            while not done:
                action = P_star[state]
                next_state, reward, done = ENV.step(action)
    
                if action == 1:
                    y = reward + ALPHA*GAMMA*MODEL(process(next_state))
                else:
                    y = np.array(reward).reshape(1,1)
                
                y_hat = MODEL.net_forward(process(state))
                
                loss_history.append(loss(y, y_hat, ALPHA))
                grad_values.append(MODEL.net_backward(y, y_hat, ALPHA))
                
                state = next_state
    
        lr = 0.001
        MODEL.batch_update_wb(lr, grad_values)
        
    
    V = dict((k,ALPHA*MODEL(process(k)).item()) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    
    return V, loss_history

#model.reset_params()
V, loss_history = train()
plot_loss(loss_history)
