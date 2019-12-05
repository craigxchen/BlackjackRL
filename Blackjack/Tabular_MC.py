import blackjack_plot_tools as bpt
import numpy as np
import gym
import pickle
from collections import defaultdict

class MCAgent:
    def __init__(self, environment, policy, gamma=1.0):
        self.gamma = gamma
        self.env = environment
        self.p = policy
        
        '''
        initialize q-table - defaultdict initializes an array with each new entry following the template of the input function
        initialize n-table - number of times we've been to a state and taken some action
        initialize v-table - max(Q(s,a))
        '''
        self.q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.n = defaultdict(lambda: 0)
        self.v = {}
    
    def update_q(self, recent_path):
        '''
        updates Q(s,a) based on the most recent experience
        alpha dictates how much we weight the recent experience
        '''
        states, actions, rewards = zip(*recent_path)
    
        for idx, s in enumerate(states):
            first_occurence_idx = next(i for i,x in enumerate(recent_path) if x[0] == s)
            
            G = sum([r*(self.gamma**i) for i,r in enumerate(rewards[first_occurence_idx:])])
            self.n[s] += 1
            
            alpha = max(1/(1+self.n[s])**0.85, 0.001)
            
            prev_q = self.q[s][actions[idx]]
            self.q[s][actions[idx]] = (1 - alpha)*prev_q + alpha*G # monte carlo
        return self.q
    
    def run(self):
        '''
        Run through the "gym" once using inputted policy
        '''
        path = [] 
        state = self.env.reset()
        while True:
            action = self.p[state]
            
            next_state, reward, done, _ = self.env.step(action)
            path.append([state, action, reward])
            state = next_state
            if done:
                break
        return path

    def test(self, policy, num_games = 1, output_details = False):
        num_wins = 0
        num_draws = 0
        for game in range(1, num_games+1):
            path = [] 
            state = self.env.reset()
            done = False
            while not done:
                action = policy[state]
                state, reward, done, _ = self.env.step(int(action))
                path.append([state, action, reward])
                
#            print("Game: {}/{}".format(game, num_games))
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

    def get_policy(self):
        return dict((k,np.argmax(v)) for k, v in self.q.items())
    
    
# %%
with open("input_policy", 'rb') as f:
    P_star = pickle.load(f)  

env = gym.make("Blackjack-v0")
num_trials = 500000
model = MCAgent(env, P_star)

for idx in range(1, num_trials+1):
    if idx % (num_trials/10) == 0:
        print("Completed: {}/{} episodes".format(idx, num_trials))
    
    path = model.run()
    model.update_q(path)
    
P_derived = model.get_policy()
model.test(P_derived, num_games=100000, output_details=False)

Q = model.q
V = dict((k,np.max(v)) for k, v in Q.items())

bpt.plot_policy(P_derived)
bpt.plot_policy(P_derived, True)
bpt.plot_v(V)
bpt.plot_v(V, True)

# final win-rate is ~40%
    