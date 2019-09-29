from blackjack_complete import CompleteBlackjackEnv
from collections import defaultdict
import numpy as np
import random
import pickle

class QLearningAgent:
    """
    A class used to group together common methods needed in Q-Learning
    
    ...
    
    Attributes
    ----------
    env : Object
        The environment the agent will interact with - in this case, simplified blackjack
        
    state_space : list
        A list of tuples that describe all possible states of blackjack

    epsilon : float
        Exploration rate

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
        self.state_space = self.env.state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.size))
        self.N = defaultdict(lambda: 0)
        
        
    def learn(self, state, action, reward, next_state):
        """Updates Q-table using standard Q-learning algorithm
        
        First, we increase the number of times we have visited a state 
        Then, we re-calculate the learning rate (alpha) using the result from Even-Dar and Mansour (JMLR 2003)
        Last, we update the Q-table: 
        if the state is a terminal state, we simply take into account the reward
        else, we default to the standard Q-learning algorithm
        
        Parameters
        ----------
        state : tuple
            tuple containing (player's sum, dealer's show card, if ace is usable)
        
        action : integer
            0 = stand, 1 = hit
        
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
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])
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
            if isinstance(policy, dict):
                action = policy[state]
            else:
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
            elif 0.3*num_iterations <= idx < 0.9*num_iterations:
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
    
    def test(self, policy, num_games = 1, output_details = False):
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
        
        output_details : bool, optional
            whether or not to output details about preceedings of each game, default = False
        """
        
        num_wins = 0
        num_draws = 0
        for game in range(num_games):
            path = self.play(policy)
            for idx, state in enumerate(path, start = 1):
                if output_details:
                    print("Your sum: {}, Dealer showed: {}, Usable Ace? {}".format(state[0][0], state[0][1], state[0][2]))
                    if state[1]:
                        print("Your action: {}".format("Hit"))
                    else: 
                        print("Your action: {}".format("Stay"))
                    if idx == len(path):
                        if state[2] == 1:
                            num_wins += 1
                            print("You won!")
                        elif state[2] == -1:
                            print("You lost.")
                        else: 
                            print("You drew.")
                            num_draws += 1
                else:
                    if idx == len(path):
                        if state[2] == 1:
                            num_wins += 1
                        elif state[2] == 0:
                            num_draws += 1
        print("Win Rate: {}/{} Draw Rate: {}/{}".format(num_wins, num_games, num_draws, num_games))
        print("Win Percentage: {:.2f}% Win+Draw: {:.2f}%".format(num_wins/num_games*100, (num_wins+(num_draws/2))/num_games*100))
        return 
    
#    def savePolicy(self, filename="blackjack_policy"):
#        with open(filename, 'wb+') as f:
#            pickle.dump(self.player_Q_Values, f)
#
#    def loadPolicy(self, filename="blackjack_policy"):
#        with open(filename, 'rb') as f:
#            self.player_Q_Values = pickle.load(f)
    
# %%
            
if __name__ == '__main__':
    env = CompleteBlackjackEnv()
    agent = QLearningAgent(env)
    
    agent.train()
    
    Q = agent.Q
    P = dict((k,np.argmax(v)) for k, v in Q.items())
    agent.test(P, 100000)
