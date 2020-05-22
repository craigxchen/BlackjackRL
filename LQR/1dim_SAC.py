"""
energy-based policy, quadratic value function
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import lqr_control as control

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def simulate_SAC(A,B,SAC,x0,T):
    """
    simulate trajectory based on policy learned by agent
    """
    x_data = []
    u_data = []
    x = x0
    u = -torch.FloatTensor(x).to(device) * SAC.agent[0].weight[:,0] / SAC.agent[0].weight[:,1]
    
    for t in range(T):
        u_data.append(u.item())
        x_data.append(x.item())
        
        u = -torch.FloatTensor(x).to(device) * SAC.agent[0].weight[:,0] / SAC.agent[0].weight[:,1]
        x = A@x + B@u.detach().numpy()
        
    return x_data, u_data

def compare_paths(x_sim,x_star,ylabel):
    fig, ax = plt.subplots()
    colors = [ '#2D328F', '#F15C19' ] # blue, orange
    
    t = np.arange(0,x_star.shape[0])
    ax.plot(t,x_sim,color=colors[0],label='Agent')
    ax.plot(t,x_star,color=colors[1],label='True')
    
    ax.set_xlabel('Time',fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    plt.legend(fontsize=18)
    
    plt.grid(True)
    plt.show()
    return

def compare_V(critic,A,B,Q,R,K,T,gamma,alpha,low=-1,high=1):
    fig, ax = plt.subplots()
    colors = [ '#B53737', '#2D328F' ] # red, blue
    label_fontsize = 18

    states = torch.linspace(low,high).detach().reshape(100,1)
    values = alpha*critic(states).squeeze().detach().numpy()

    ax.plot(states.numpy(),values,color=colors[0],label='Approx. Loss Function')
    ax.plot(states.numpy(),control.trueloss(A,B,Q,R,K,states.numpy(),T,gamma).reshape(states.shape[0]),color=colors[1],label='Real Loss Function')


    ax.set_xlabel('x',fontsize=label_fontsize)
    ax.set_ylabel('y',fontsize=label_fontsize)
    plt.legend()

    plt.grid(True)
    plt.show()
    return

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

# "custom" activation function for pytorch - compatible with autograd
class Quadratic(nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()

    def forward(self, x):
        return x**2

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, tau):
        super(Model, self).__init__()

        self.agent = nn.Sequential(
                # Quadratic(),
                nn.Linear(state_dim + action_dim, n_latent_var, bias=False),
                Quadratic(),
                # nn.Linear(n_latent_var, 1, bias=False)
                )
        
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = -state * self.agent[0].weight[:,0] / self.agent[0].weight[:,1]
        
        # action_var = torch.full((action_dim,), 0.5 * self.tau)
        action_var = 0.5 * self.tau / self.agent[0].weight[:,1]*self.agent[0].weight[:,1]
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
        action_means = -states * self.agent[0].weight[:,0] / self.agent[0].weight[:,1]
        
        # action_var = torch.full((action_dim,), 0.5 * self.tau)
        action_var = 0.5 * self.tau / self.agent[0].weight[:,1]*self.agent[0].weight[:,1]
        action_var = action_var.expand_as(action_means)
        
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_means, cov_mat)
        
        action_logprobs = dist.log_prob(actions)
        action_values = self.agent(torch.cat((states,actions),dim=1).squeeze())
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(action_values), dist_entropy
        

class SAC:
    def __init__(self, state_dim, action_dim, n_latent_var, tau, lr, betas, gamma, K_epochs):
        self.betas = betas
        self.gamma = gamma
        self.tau = tau
        self.K_epochs = K_epochs
        
        self.policy = Model(state_dim, action_dim, n_latent_var, tau).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.agent.parameters(), lr=lr, betas=betas)
        
        self.policy_old = Model(state_dim, action_dim, n_latent_var, tau).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
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
            logprobs, action_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                
            # Finding Loss:
            actor_loss = action_values - logprobs
            critic_loss = 0.5 * self.MseLoss(action_values , costs)
            loss = actor_loss + critic_loss - dist_entropy / self.tau
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
    
############## Hyperparameters ##############

A = np.array(1).reshape(1,1)
B = np.array(1).reshape(1,1)
Q = np.array(1).reshape(1,1)
R = np.array(1).reshape(1,1)

state_dim = 1
action_dim = 1
log_interval = 100            # print avg cost in the interval
max_episodes = 100000         # max training episodes
max_timesteps = 10            # max timesteps in one episode

solved_cost = None

n_latent_var = 1             # number of variables in hidden layer
tau = 1.00                   # temperature constant
K_epochs = 1                 # update policy for K epochs
gamma = 1.00                 # discount factor
                         
lr = 0.01        
betas = (0.9, 0.999)         # parameters for Adam optimizer

random_seed = 1
#############################################

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
memory = Memory()
sac = SAC(state_dim, action_dim, n_latent_var, tau, lr, betas, gamma, K_epochs)
print(f"device: {device}, lr: {lr}, tau: {tau}, betas: {betas}")  

# logging variables
running_cost = 0

# training loop
for i_episode in range(1, max_episodes+1):
    state = 5*np.random.randn(1,1)
    done = False
    for t in range(max_timesteps):
        # Running policy_old:
        action = sac.select_action(state, memory)
        
        cost = state@Q@state + np.array(action).reshape(1, 1)@R@np.array(action).reshape(1,1)
        state = A@state + B@np.array(action).reshape(1,1)

        # Saving cost and is_terminals:
        memory.costs.append(cost.item())
        memory.is_terminals.append(done)
        
        if done:
            break
        
    sac.update(memory)

    memory.clear_memory()
        
    running_cost += cost.item()
        
    # logging
    if i_episode % log_interval == 0:        
        # print(list(sac.policy.agent.parameters())[0].grad)
        
        print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost/log_interval))
        running_cost = 0
        
        
# random init to compare how the two controls act
x0 = np.random.uniform(-5,5,(1,))
u0 = np.zeros((1,))
T = 50

# Optimal control for comparison
K, P, _ = control.dlqr(A,B,Q,R)

x_star, u_star = control.simulate_discrete(A,B,K,x0.reshape(1,1),u0.reshape(1,1),T)
x_sim, u_sim = simulate_SAC(A,B,sac.policy,x0,T)

compare_paths(np.array(x_sim), np.squeeze(x_star[:,:-1]), "state")
compare_paths(np.array(u_sim), np.squeeze(u_star[:,:-1]), "action")
