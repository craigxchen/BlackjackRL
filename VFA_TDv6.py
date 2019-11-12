from VFA_Net import NeuralNetwork
from blackjack_complete_TEST import CompleteBlackjackEnv
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import numpy as np

# try last-layer zeros or doubling
# lowering step size
# increasing alpha
# try batch update

"""
test results:
    
    using ALPHA = 1000, GAMMA = 1,
    NUM_TRIALS = 100000 and 500000
    512 neurons in hidden layer
    
    function initialized to zero
    
    converges when:
        1-hot encoding and relu, leakyRelu
        normalized vector encoding and _
"""


nn_arq = [
    {"input_dim": 3, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 1, "activation": "none"},
]

ALPHA = 1000
GAMMA = 1
NUM_TRIALS = 500000

def loss(target, prediction, alpha=1):
    return float((1/(alpha**2))*np.square(target-alpha*prediction))

model = NeuralNetwork(nn_arq, bias = True, double = "yes")   
env = CompleteBlackjackEnv()

# %% 

def process(state):
    # uncomment return statements for different types of feature mappings, 
    # don't forget to change the input dim of the neural net
    
    '''normalized vector''' 
    return np.array([state[0]/(max(env.state_space)[0]), state[1]/(max(env.state_space)[1]), state[2]]).reshape((3,1))
    '''stretched vector'''
#    return np.array([state[0]*10, state[1]*10, state[2]*10]).reshape(3,1)
    '''sum all'''
#    return np.array(np.sum([state[0], state[1], state[2]])).reshape((1,1))
    '''standard vector'''
#    return np.array([state[0], state[1], state[2]]).reshape((3,1))
    '''one-hot'''
#    return np.array([int(state == k) for k in env.state_space]).reshape(len(env.state_space),1)

def get_policy(V):
    P = {}
    for state in V.keys():
        EV_Hit = []
        EV_Stay = model(process(state))
        
        fut_states, _ = env.future_states(state)
#        fut_states = [f for f in fut_states if f[0] <= 21]
        for fs in fut_states:
            EV_Hit.append(model(process(fs)))

        P[state] = (np.mean(EV_Hit) > EV_Stay)[0][0].astype(int)
    return P

def plot_v(V, usable_ace = False):
        fig, ax = plt.subplots()
        ax = Axes3D(fig)
        
        states = list(V.keys())
        states_YESace = {}
        states_NOTace = {}
        for state in states:
            if not state[2]:
                states_NOTace[state] = V[state]
            else: 
                states_YESace[state] = V[state]
        
        if usable_ace == 1:
            player_sum = [state[0] for state in states_YESace.keys()]
            dealer_show = [state[1] for state in states_YESace.keys()]
            scores = [val for val in states_YESace.values()]

            ax.plot_trisurf(player_sum, dealer_show, scores, cmap="viridis", edgecolor="none")
            ax.set_xlabel("Player's Sum")
            ax.set_ylabel("Dealer's Show Card")
            ax.set_zlabel("Perceived Value")
            ax.set_title("Soft Sums")
            ax.view_init(elev=40, azim=-100)
        else:
            player_sum = np.array([state[0] for state in states_NOTace.keys()])
            dealer_show = np.array([state[1] for state in states_NOTace.keys()])
            scores = np.array([val for val in states_NOTace.values()])

            ax.plot_trisurf(player_sum, dealer_show, scores, cmap="viridis", edgecolor="none")
            ax.set_xlabel("Player's Sum")
            ax.set_ylabel("Dealer's Show Card")
            ax.set_zlabel("Perceived Value")
            ax.set_title("Hard Sums")
            ax.view_init(elev=40, azim=-100)
        return

def plot_policy(policy, usable_ace = False, save = True):
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
    
    # create discrete colormaps
    cmap = colors.ListedColormap(['red', 'green'])
    bounds = [-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    
    ax.set_xticks(np.arange(10))
    if not usable_ace:
        ax.set_yticks(np.arange(18))
        ax.set_yticklabels(['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
                            '14', '15', '16', '17', '18', '19', '20', '21'])
    else:
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(['A + 1', 'A + 2', 'A + 3', 'A + 4', 'A + 5', 'A + 6', 
                            'A + 7', 'A + 8', 'A + 9', 'A + 10'])
        
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
    
    # TODO Fix
    if save:
        if usable_ace:
            fig.savefig("VFA_policy_soft({}trials,{}alpha,{}learningrate,{}neurons).png"
                        .format(NUM_TRIALS,ALPHA,"0.001/ALPHA",nn_arq[0]["output_dim"]), bbox_inches = 'tight')
        else:
            fig.savefig("VFA_policy_hard({}trials,{}alpha,{}learningrate,{}neurons).png"
                        .format(NUM_TRIALS,ALPHA,"0.001/ALPHA",nn_arq[0]["output_dim"]), bbox_inches = 'tight')
    return

def plot_loss(y):
    fig, ax = plt.subplots()
    label_fontsize = 18

    t = np.arange(0,len(y))
    ax.plot(t[::int(NUM_TRIALS/500)],y[::int(NUM_TRIALS/500)])
        
    ax.set_yscale('log')
    ax.set_xlabel('Trials',fontsize=label_fontsize)
    ax.set_ylabel('Loss',fontsize=label_fontsize)

    plt.grid(True)
    plt.show()
    return

# %% training
with open("near_optimal", 'rb') as f:
    P_star = pickle.load(f)    

def train(**kwargs):
    loss_history = []
    for i in range(NUM_TRIALS):
        if (i+1)%(NUM_TRIALS/10) == 0:
            print('trial {}/{}'.format(i+1,NUM_TRIALS))
            
        state, _ = env.reset()
        done = False
        
        while not done:
            action = P_star[state]
            next_state, reward, done = env.step(action)

            if action == 1:
                fut_vals = []
                fut_states, fut_rewards = env.future_states(state)
                for s,r in zip(fut_states, fut_rewards):
                    fut_vals.append(r + ALPHA*GAMMA*model(process(s)))
                
                y = np.mean(fut_vals)
#                y = reward + ALPHA*GAMMA*model(process(next_state))
            else:
                y = reward
            
            y_hat = model.net_forward(process(state))
            
            lr = 0.001
            
            loss_history.append(loss(y, y_hat, ALPHA))
            model.net_backward(y, y_hat, ALPHA)
            model.update_wb(lr)
            
            state = next_state   
        
    
    V = dict((k,model(process(k)).item()) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    P_derived = get_policy(V)
    
    return P_derived, V, loss_history

#model.reset_params()
P_derived, V, loss_history = train()
plot_loss(loss_history)
plot_policy(P_derived, save=False)
plot_policy(P_derived, True, save=False)
plot_v(V)
plot_v(V, True)
