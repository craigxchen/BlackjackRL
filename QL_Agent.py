from blackjack_complete import CompleteBlackjackEnv
from collections import defaultdict
import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, environment, epsilon = 1.0, gamma = 1.0):
        self.env = environment
        self.state_space = [(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = {}
        for s in self.state_space:
            if s[0] == 21:
                self.Q[s] = [1, -1]
            else:
                self.Q[s] = [0, 0]
        self.N = defaultdict(lambda: 0)
        
        
    def learn(self, state, action, reward, next_state):
        '''
        updates Q-table using standard Q-learning algorithm
        '''
        self.N[state] += 1
        alpha = 1/(1 + self.N[state])**0.85
        if state == next_state:
            self.Q[state][action] = (1 - alpha)*self.Q[state][action] + alpha*reward
        else:
            self.Q[state][action] = (1 - alpha)*self.Q[state][action] + alpha*(reward + self.gamma*np.max(self.Q[next_state]))
        return
    
    def policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def play(self):

        # start
        state = self.env.reset()
        history = []
        done = False
        
        while not done:
            action = self.policy(state)
            next_state, reward, done = self.env.step(action)
            history.append([state, action, reward, done])
            state = next_state

        return history
    
    def train(self, num_iterations = 100000):
        for idx in range(num_iterations):
            if idx < 0.3*num_iterations:
                self.epsilon *= 0.999995
                #self.alpha *= 0.999995
            elif 0.3*num_iterations <= idx < 0.9*num_iterations:
                self.epsilon *= 0.99995
                #self.alpha *= 0.99995
            else: 
                self.epsilon = 0
                #self.alpha = 0
            episode = self.play()
            states, actions, rewards, done = zip(*episode)
            for k in range(len(states)):
                if k == len(states) - 1:
                    self.learn(states[k], actions[k], rewards[k], states[k])
                else: 
                    self.learn(states[k], actions[k], rewards[k], states[k+1])
    
        return
    
    def test(self, policy, num_games = 1, output_details = False):
        num_wins = 0
        num_draws = 0
        for game in range(1, num_games+1):
            path = self.play()
            if game == num_games % num_games/10:
                print("Game: {}/{}".format(game, num_games))
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
    
    def savePolicy(self, filename="blackjack_policy"):
        with open(filename, 'wb+') as f:
            pickle.dump(self.player_Q_Values, f)

    def loadPolicy(self, filename="blackjack_policy"):
        with open(filename, 'rb') as f:
            self.player_Q_Values = pickle.load(f)
    
# %%
            
if __name__ == '__main__':
    env = CompleteBlackjackEnv()
    agent = QLearningAgent(env)
    
    agent.train()
    
    Q = agent.Q
    P = dict((k,np.argmax(v)) for k, v in Q.items())
    agent.test(P, 100000)