'''
Blackjack environment for reinforcement learning agent
'''

import numpy as np
import random
import gym

class BlackjackSM:
    def __init__(self, epsilon = 1.0, alpha = 0.1, gamma = 1.0):
        self.env = gym.make("Blackjack-v0")
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

    def policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def learn(self, state, action, reward, next_state):
        if state == next_state:
            self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*reward
        else:
            self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*np.max(self.Q[next_state]))
        return
    
    def play(self):
        history = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action) 
            history.append([state, action, reward, done])
            state = next_state
        return history

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
        print("Win Percentage: {:.2f}% Win+Draw: {:.2f}%".format(num_wins/num_games*100, (num_wins+num_draws)/num_games*100))
        return
    
#    def dealer_policy(self):
#        d_cards = [self.deal_card(), self.deal_card()]
#        show_card = d_cards[1]
#        d_sum = np.sum(d_cards)
#        if d_cards.count(1) >= 1:
#            usable_ace = True
#        if d_sum > 17:
#            done = True
#        else: 
#            d_cards.append(self.deal_card())
#        return
#    
    
    @staticmethod
    def deal_card():
        return random.choice(list(range(1,11) + 3*[10]))      
    
# %% Main
        
if __name__ == '__main__':
    model = BlackjackSM()
    num_iterations = 100000
    for idx in range(num_iterations):
        if idx < 0.3*num_iterations:
            model.epsilon *= 0.999995
        elif 0.3*num_iterations <= idx < 0.9*num_iterations:
            model.epsilon *= 0.99995
        else: 
            model.epsilon = 0
            model.alpha = 0
        game = model.play()
        states, actions, rewards, done = zip(*game)
        for k in range(len(states)):
            if k == len(states) - 1:
                model.learn(states[k], actions[k], rewards[k], states[k])
            else: 
                model.learn(states[k], actions[k], rewards[k], states[k+1])
                
    Q = model.Q
    P = dict((k,np.argmax(v)) for k, v in Q.items())
    model.test(P, 100000)
