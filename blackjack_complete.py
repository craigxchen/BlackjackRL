'''
Blackjack environment for reinforcement learning agent
'''

import numpy as np
import pickle
import random

class CompleteBlackjackEnv:
    def __init__(self):
        self.action_space = np.array([0, 1])
        self.state_space = [(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]
        
        
    def step(self, action):
        assert action in self.action_space
        if action:  # hit: add a card to players hand and return
            self.player.append(self.deal_card())
            if self.is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            # hit 17 strategy
            while self.sum_hand(self.dealer) <= 17:
                self.dealer.append(self.deal_card())
            reward = self.winner(self.score(self.player), self.score(self.dealer))
        return self.get_playerstate(), reward, done 
    
    def reset(self):
        '''
        also functions as start
        '''
        self.player = [self.deal_card(), self.deal_card()]
        self.dealer = [self.deal_card(), self.deal_card()]
        
        return self.get_playerstate()
    
    @staticmethod
    def winner(player, dealer):
        return (player > dealer) - (dealer > player)
    
    @staticmethod
    def deal_card():
        return random.choice(list(range(1,11)) + 3*[10])

    def get_playerstate(self):
        return (self.sum_hand(self.player), self.dealer[0], self.usable_ace(self.player))
    
    @staticmethod
    def usable_ace(hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21
    
    def sum_hand(self, hand):  # Return current hand total
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)
    
    def is_bust(self, hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21
    
    def score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)


# %%
class QLearningAgent:
    def __init__(self, environment, epsilon = 1.0, alpha = 0.1, gamma = 1.0):
        self.env = environment
        self.state_space = [(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        for s in self.state_space:
            if s[0] == 21:
                self.Q[s] = [1, -1]
            else:
                self.Q[s] = [0, 0]
        
    def learn(self, state, action, reward, next_state):
        '''
        updates Q-table using standard Q-learning algorithm
        '''
        if state == next_state:
            self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*reward
        else:
            self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*np.max(self.Q[next_state]))
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
            elif 0.3*num_iterations <= idx < 0.9*num_iterations:
                self.epsilon *= 0.99995
            else: 
                self.epsilon = 0
                self.alpha = 0
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
    blackjack = CompleteBlackjackEnv()
    agent = QLearningAgent(blackjack)
    
    agent.train()
    
    Q = agent.Q
    P = dict((k,np.argmax(v)) for k, v in Q.items())
    agent.test(P, 100000)
