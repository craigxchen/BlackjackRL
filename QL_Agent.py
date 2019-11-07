from blackjack_complete import CompleteBlackjackEnv
from collections import defaultdict
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import pickle

class QLAgent:
    """
    A class used to group together common methods needed in Q-Learning
    
    ...
    
    Attributes
    ----------
    env : Object
        The environment the agent will interact with - in this case, simplified blackjack

    tau : float
        Temperature for softmax 

    gamma : float
        Discount rate

    Q : dict
        Keys: State tuples
        Values: list [perceived value of action = 0, perceived value of action = 1]
        
    N : dict
        keys: State tuples
        Values: number of times the state has been visited
    
    Methods
    -------
    learn(state, action, reward, next state)
        Updates Q-table using standard Q-learning algorithm
        
    policy(state)
        Chooses action based on state
        
    play()
        Goes through one full iteraction with the environment
        
    train(number of iterations)
        Trains the agent to learn the optimal Q-table
    
    test(policy, number of games = 1, output details = False):
        Evaluates a certain policy
    """
    
    def __init__(self, environment, epsilon = 1.0, gamma = 1.0):
        """
        Parameters
        ----------
        env : Object
            The environment the agent will interact with - in this case, simplified blackjack
        
        epsilon : float, optional
            The rate at which the agent will explore - decays over time (default is 1)
            Values range from 0 - 1
            
        gamma : float, optional
            The discount rate we apply to future rewards (default is 1)
        """
        
        self.env = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        self.N = defaultdict(lambda: 0)
        
        
    def learn(self, state, action, reward, next_state):
        """Updates Q-table using standard Q-learning algorithm
        
        First, we increase the number of times we have visited a state 
        Then, we re-calculate the learning rate (alpha) using the result from Even-Dar and Mansour (JMLR 2003)
        Last, we update the Q-table: 
        if the state is a terminal state, we simply take into account the reward
        else, we default to the standard Q-learning algorithm
        
        States are updated after completion of an episode because episodes tend to be 1,2 time-steps each
        Additionally, after leaving a state, it is very unlikely to return to that state during one episode
        
        Parameters
        ----------
        state : tuple
            tuple containing (player's sum, dealer's show card, if ace is usable)
        
        action : integer
        
        reward : float
            reward at a state
        
        next_state : tuple
            see 'state'
        """
        
        self.N[state] += 1
        alpha = 1/(1 + self.N[state])**0.85
        if state == next_state:
            self.Q[state][action] = (1 - alpha)*self.Q[state][action] + alpha*reward
        else:
            self.Q[state][action] = (1 - alpha)*self.Q[state][action] + alpha*(reward + self.gamma*np.max(self.Q[next_state]))
        return
    
    def policy(self, state):
        """Chooses action based on state
        
        (epsilon*100) % of the time, we pick a random action to "explore"
        The rest of the time, we pick the action that corresponds to the maximum future reward
        
        Parameters
        ----------
        state : tuple
            tuple containing (player's sum, dealer's show card, if ace is usable)
            
        Returns
        -------
        action : int
            0 or 1 (i.e. stand or hit)
        """
        #decaying epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.env.action_space)
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def play(self, policy):
        """Goes through one full iteraction with the environment
        
        Runs through a full iteraction with the environment following a specific policy
        Records the result of the MDP in a list of lists history
        
        Returns
        -------
        history : list
            ordered list of lists that contain state, action, reward, done information
        """
        
        # start
        state = self.env.reset()
        history = []
        done = False
        
        while not done:
            # if policy is a dictionary
            if isinstance(policy, dict):
                action = policy[state]
            else: # policy is a function
                action = policy(state)
            next_state, reward, done = self.env.step(action)
            history.append([state, action, reward, done])
            state = next_state

        return history
    
    def train(self, num_iterations = 100000):
        """Trains the agent to learn the optimal Q-table
        
        Over 100,000 traisl, the exploration rate, epsilon, slowly decays over time
        decays very slowly in the beginning, at a more rapid rate during the middle, and then drops to 0 near the end
        Each iteration, we experience an iteraction with the environment, and then loop through the history to learn from it
        
        Parameters
        ----------
        num_iterations : int, optional
            number of training rounds, defaults to 100,000
        """
        
        for idx in range(num_iterations):
            if idx < 0.3*num_iterations:
                self.epsilon *= 0.999995
            elif 0.3*num_iterations <= idx < 0.8*num_iterations:
                self.epsilon *= 0.99995
            else: 
                self.epsilon = 0
            episode = self.play(self.policy)
            states, actions, rewards, done = zip(*episode)
            for k in range(len(states)):
                if k == len(states) - 1: # if states[k] is the terminal state
                    self.learn(states[k], actions[k], rewards[k], states[k])
                else: 
                    self.learn(states[k], actions[k], rewards[k], states[k+1])
    
        return
    
    def test(self, policy, num_games = 10):
        """Evaluates a certain policy
        
        Over 100,000 traisl, the exploration rate, epsilon, slowly decays over time
        decays very slowly in the beginning, at a more rapid rate during the middle, and then drops to 0 near the end
        Each iteration, we experience an iteraction with the environment, and then loop through the history to learn from it
        Prints out the win rate, draw rate, and the corresponding percentages
        
        Parameters
        ----------
        policy : dict
            Dictionary of state, action pairs that controls how the agent behaves
        
        num_games : int, optional
            number of games to play, default = 1
        """
        
        num_wins = 0
        num_draws = 0
        for game in range(num_games):
            path = self.play(policy)
            if path[-1][2] == 1 or path[-1][2] == 2:
                num_wins += 1
            elif path[-1][2] == 0:
                num_draws += 1
        print("Win Rate: {}/{} Draw Rate: {}/{}".format(num_wins, num_games, num_draws, num_games))
        print("Win Percentage: {:.2f}% Win+Draw: {:.2f}%".format(num_wins/num_games*100, (num_wins+(num_draws/2))/num_games*100))
        return 
    
    def save_policy(self, filename="blackjack_policy"):
        with open(filename, 'wb+') as f:
            pickle.dump(self.player_Q_Values, f)

    def load_policy(self, filename="blackjack_policy"):
        with open(filename, 'rb') as f:
            self.player_Q_Values = pickle.load(f)
    
# %%
            
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
    cmap = colors.ListedColormap(['red', 'green', 'blue'])
    bounds = [-0.5,0.5,1.5,2.5]
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
        
    ax.set_xticklabels(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'])
    
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

if __name__ == '__main__':
    env = CompleteBlackjackEnv()
    agent = QLAgent(env)
    
    agent.train(1000000)
    
    Q = agent.Q
    P = dict((k,np.argmax(v)) for k, v in Q.items())
    V = dict((k,np.max(v)) for k,v in Q.items())
    agent.test(P, 100000)
    plot_policy(P)
    plot_policy(P, True)
    plot_v(V)
    plot_v(V, True)
    
