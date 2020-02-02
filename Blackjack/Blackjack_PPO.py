"""
MIT License

Copyright (c) 2018 Nikhil Barhate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Modified by Craig Chen 2020

"""


import sys, os
sys.path.append(os.path.abspath(os.path.join("..\Blackjack")))
import blackjack_plot_tools as bpt

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, zero=False, double=False):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, 1)
                )
        
        if zero:
            with torch.no_grad():
                self.actor[-2].weight = nn.Parameter(torch.zeros([action_dim, n_latent_var]))
                if self.actor[-2].bias is not None:
                    self.actor[-2].bias = nn.Parameter(torch.zeros([action_dim,]))
                
                self.critic[-1].weight = nn.Parameter(torch.zeros([1, n_latent_var]))
                if self.critic[-1].bias is not None:
                    self.critic[-1].bias = nn.Parameter(torch.zeros([1,]))
                    
        if double:
            with torch.no_grad():
                temp1 = torch.randn([n_latent_var//2,state_dim]) * np.sqrt(2/n_latent_var)
                self.critic[0].weight = nn.Parameter(torch.cat((temp1,temp1),dim=0))
                
                temp2 = torch.randn([1,n_latent_var//2]) * np.sqrt(2/n_latent_var)
                self.critic[-1].weight = nn.Parameter(torch.cat((temp2,-temp2),dim=1))
                
                temp3 = torch.randn([n_latent_var//2,state_dim]) * np.sqrt(2/n_latent_var)
                self.actor[0].weight = nn.Parameter(torch.cat((temp3,temp3),dim=0))
                
                temp4 = torch.randn([action_dim,n_latent_var//2]) * np.sqrt(2/n_latent_var)
                self.actor[-2].weight = nn.Parameter(torch.cat((temp4,-temp4),dim=1))
    
    def get_dist(self, state):
        state = torch.from_numpy(np.array(state)).float().to(device) 
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        return dist
    
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(np.array(state)).float().to(device) 
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        if not memory:
            return action.item()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, actor_lr, critic_lr, betas, gamma, K_epochs, eps_clip, zero=False, double=False):
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, zero, double).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=actor_lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=critic_lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, zero, double).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update_actor(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss: see https://spinningup.openai.com/en/latest/algorithms/ppo.html
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (torch.ones_like(advantages) + self.eps_clip * torch.sign(advantages)) * advantages
#            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            
            # take gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def update_critic(self, memory, alpha=100):   
        states = torch.stack(memory.states).to(device).detach()
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing and scaling the rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = alpha*(rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize critic for K epochs:
        for _ in range(self.K_epochs):
            state_values = alpha*torch.squeeze(self.policy.critic(states))
            
            critic_loss = 0.5/(alpha**2)*self.MseLoss(state_values,rewards)
            
            # take gradient step
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()

            
if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = "Blackjack-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = 3
    action_dim = 2
    log_interval = 5000         # print avg reward in the interval
    max_episodes = 250000       # max training episodes
    batch_size = 64
    n_latent_var = 256          # number of variables in hidden layer
    update_timestep = 128       # update policy every n timesteps
    
    reward_threshold = -0.04    # if avg reward > threshold, break out of training
    
    actor_lr = 0.0003
    critic_lr = 0.001
    alpha = 100
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 5                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, actor_lr, critic_lr, betas, gamma, K_epochs, eps_clip, zero=False, double=True)
    print("actor lr: {}, critic lr: {}, betas: {}".format(actor_lr,critic_lr,betas))
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(batch_size):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            next_state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update_actor(memory)
                ppo.update_critic(memory, alpha)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            state = next_state
            
            if done:
                break
                
        avg_length += t
        
        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length/log_interval
            running_reward = running_reward/log_interval
            
            if running_reward > reward_threshold:
                break
            
            print('Episode {} \t avg length: {:0.2f} \t reward: {:0.2f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
            
    P = dict((k,ppo.policy.act(np.array(k),None)) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    V = dict((k,ppo.policy.critic(torch.from_numpy(np.array(k)).float().to(device).detach()).item()) for k in ([(x, y, True) for x in range(12,22) for y in range(1,11)] + [(x, y, False) for x in range(4,22) for y in range(1, 11)]))
    bpt.plot_policy(P, False)
    bpt.plot_policy(P, True)
    bpt.plot_v(V, False)
    bpt.plot_v(V, True)
