import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import lqr_control as control

# temp fix for OpenMP issue
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def simulate(A,B,policy,x0,T):
    """
    simulate trajectory based on policy learned by agent
    """
    x_data = []
    u_data = []
    x = x0
    u = policy(torch.FloatTensor(x.reshape(1, -1)).to(device)).detach()
    
    for t in range(T):
        u_data.append(u.item())
        x_data.append(x.item())
        
        u = policy(torch.FloatTensor(x.reshape(1, -1)).to(device)).detach()
        x = A@x + B@u.numpy()
        
    return x_data, u_data

def compare_paths(x_sim,x_star,ylabel):
    fig, ax = plt.subplots()
    colors = [ '#2D328F', '#F15C19' ] # blue, orange
    
    t = np.arange(0,x_star.shape[0])
    ax.plot(t,x_star,color=colors[1],label='True')
    ax.plot(t,x_sim,color=colors[0],label='Agent')
    
    ax.set_xlabel('Time',fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    plt.legend(fontsize=18)
    
    plt.grid(True)
    plt.show()
    return

# "custom" activation functions for pytorch - compatible with autograd
class Spike(nn.Module):
    def __init__(self, center=1, width=1):
        super(Spike, self).__init__()
        self.c = center
        self.w = width
        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.alpha * torch.min(torch.max((x - (self.c - self.w)),torch.zeros_like(x)),torch.max((-x + (self.c + self.w)),torch.zeros_like(x)))
        

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.costs = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.costs[:]
        del self.is_terminals[:]

class PRELU(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, sigma):
        super(PRELU, self).__init__()

        self.agent = nn.Sequential(
                nn.PReLU(),
                nn.Linear(state_dim, action_dim, bias=False)
                )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sigma = sigma
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.agent(state)
        
        action_var = torch.full((action_dim,), self.sigma)
        action_var = action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, states, actions):
        action_means = self.agent(states)
        
        action_var = torch.full((action_dim,), self.sigma)
        action_var = action_var.expand_as(action_means)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_means, cov_mat)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_logprobs, dist_entropy

# TODO
class CHAOS(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, sigma):
        super(CHAOS, self).__init__()

        self.agent = nn.Sequential(
                nn.Linear(state_dim, action_dim, bias=False),
                Spike(),
                # nn.Linear(n_latent_var, action_dim, bias=False)
                )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sigma = sigma
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.agent(state)
        
        action_var = torch.full((action_dim,), self.sigma)
        action_var = action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, states, actions):
        action_means = self.agent(states)
        
        action_var = torch.full((action_dim,), self.sigma)
        action_var = action_var.expand_as(action_means)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_means, cov_mat)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_logprobs, dist_entropy    


class PG:
    def __init__(self, state_dim, action_dim, n_latent_var, sigma, lr, betas, gamma, K_epochs):
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.sigma = sigma
        
        self.policy = CHAOS(state_dim, action_dim, n_latent_var, sigma).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.agent.parameters(), lr=lr, betas=betas)
        
        
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):   
        # Monte Carlo estimate of state costs:
        costs = []
        discounted_cost = 0
        for cost, is_terminal in zip(reversed(memory.costs), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_cost = 0
            discounted_cost = cost + (self.gamma * discounted_cost)
            costs.insert(0, discounted_cost)
        
        # Normalizing the costs:
        costs = torch.tensor(costs).to(device)
        # costs = (costs - costs.mean()) / (costs.std() + 1e-8)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
                
            # Finding Loss:
            actor_loss = costs * logprobs
            loss = actor_loss - self.sigma * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
    
    
############## Hyperparameters ##############

A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R = np.array(1).reshape(1,1)

state_dim = 1
action_dim = 1
log_interval = 100            # print avg cost in the interval
max_episodes = 100000         # max training episodes
max_timesteps = 10           # max timesteps in one episode

solved_cost = None

n_latent_var = 1             # number of variables in hidden layer
sigma = 0.1                  # standard deviation of actions
K_epochs = 1                 # update policy for K epochs
gamma = 0.99                 # discount factor
                         
lr = 0.0003        
betas = (0.9, 0.999)         # parameters for Adam optimizer

random_seed = None
#############################################

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
memory = Memory()
pg = PG(state_dim, action_dim, n_latent_var, sigma, lr, betas, gamma, K_epochs)
print(f"device: {device}, lr: {lr}, sigma: {sigma}, betas: {betas}")  

# logging variables
running_cost = 0

# training loop
for i_episode in range(1, max_episodes+1):
    state = 5*np.random.randn(1,1)
    done = False
    for t in range(max_timesteps):
        # Running policy_old:
        action = pg.select_action(state, memory)
        
        cost = state@Q@state + np.array(action).reshape(1, 1)@R@np.array(action).reshape(1,1)
        
        if np.random.uniform(0,1) > gamma:
            state = np.random.randn(1,1)
        else:
            state = A@state + B@np.array(action).reshape(1,1)

        # Saving cost and is_terminals:
        memory.costs.append(cost.item())
        memory.is_terminals.append(done)
        
        if done:
            break
        
    pg.update(memory)

    memory.clear_memory()
        
    running_cost += cost.item()
        
    # logging
    if i_episode % log_interval == 0:        
        print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost/log_interval))
        running_cost = 0
        
        
# random init to compare how the two controls act
x0 = 5*np.random.randn(1,1)
T = 50

# Optimal control for comparison
K, P, _ = control.dlqr(A,B,Q,R)

x_star, u_star = control.simulate_discrete(A,B,K,x0.reshape(1,1),T)
x_sim, u_sim = simulate(A,B,pg.policy.agent,x0,T)

compare_paths(np.array(x_sim), np.squeeze(x_star[:,:-1]), "state")
compare_paths(np.array(u_sim), np.squeeze(u_star[:,:-1]), "action")
