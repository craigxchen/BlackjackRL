import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import lqr_control as control

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate(A,B,policy,x0,u0,T):
    """
    simulate trajectory based on policy learned by PPO agent
    """
    x_data = []
    u_data = []
    x = x0
    u = u0
    for t in range(T):
        u_data.append(u.item())
        x_data.append(x.item())
        
        u = policy(torch.as_tensor(x).float()).detach().numpy()
        x = np.matmul(A, x) + np.matmul(B, u)
        
    return x_data, u_data

def compare_paths(x_sim,x_star):
    fig, ax = plt.subplots()
    colors = [ '#2D328F', '#F15C19' ] # blue, orange
    
    t = np.arange(0,x_star.shape[0])
    ax.plot(t,x_sim,color=colors[0],label='Agent')
    ax.plot(t,x_star,color=colors[1],label='True')
    
    ax.set_xlabel('Time',fontsize=18)
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
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# "custom" activation function for pytorch - compatible with autograd
class Quadratic(nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()

    def forward(self, x):
        return x**2

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, zero=False):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, n_latent_var, bias=False),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim, bias=False),
                nn.Tanh(),
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, n_latent_var, bias=False),
                Quadratic(),
                nn.Linear(n_latent_var, 1, bias=False)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
        if zero: 
            with torch.no_grad():                
                self.critic[-1].weight = nn.Parameter(torch.zeros([1, n_latent_var]))
#                self.critic[-1].bias = nn.Parameter(torch.zeros([1, 1]))
                
                self.actor[-1].weight = nn.Parameter(torch.zeros([action_dim, n_latent_var]))
#                self.actor[-1].bias = nn.Parameter(torch.zeros([action_dim, 1]))
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action, alpha):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = alpha*self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, alpha, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.alpha = alpha
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std, zero=True).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std, zero=True).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, self.alpha)
            
            # Finding the ratio (pi_theta / pi_theta__old): clamp the logprobs to prevent NaN's
            ratios = torch.exp(torch.clamp(logprobs - old_logprobs, -1, 1))
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (torch.ones_like(advantages) + self.eps_clip * torch.sign(advantages)) * advantages
            actor_loss = -torch.min(surr1, surr2)
            
            # take gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def update_critic(self, memory):   
        states = torch.stack(memory.states).to(device).detach()
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        rewards = self.alpha*(rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize critic for K epochs:
        for _ in range(self.K_epochs):
            state_values = self.alpha*torch.squeeze(self.policy.critic(states))
            
            critic_loss = 0.5/(self.alpha**2)*self.MseLoss(state_values,rewards)
            
            # take gradient step
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()
    
            
if __name__ == '__main__':
    ############## Hyperparameters ##############
    
    A = np.array(1).reshape(1,1)
    B = np.array(1).reshape(1,1)
    Q = np.array(-1).reshape(1,1)
    R = np.array(-1).reshape(1,1)
    
    state_dim = 1
    action_dim = 1
    log_interval = 500          # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 50          # max timesteps in one episode
    
#    solved_reward = None
    
    n_latent_var = 64           # number of variables in hidden laye
    update_timestep = 500       # update policy every n timesteps
    action_std = 0.1            # constant std for action distribution (Multivariate Normal)
    K_epochs = 5                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    alpha = 100
    lr = 0.0005                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, alpha, gamma, K_epochs, eps_clip)
    print("lr: {}, betas: {}".format(lr,betas))  
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = np.random.randn(1).reshape(1,1)
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            
            reward = np.matmul(state,np.matmul(Q,state)) + np.matmul(np.array(action).reshape(1,1),np.matmul(R,np.array(action).reshape(1,1)))
            state = np.matmul(A,state) + np.matmul(B,np.array(action).reshape(1,1))
#            print(reward,t)
            # Saving reward and is_terminals:
            memory.rewards.append(reward.item())
            memory.is_terminals.append(False)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update_actor(memory)
                ppo.update_critic(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward.item()
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
#        if running_reward > (log_interval*solved_reward):
#            print("########## Solved! ##########")
#            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format("1dim_LQR"))
#            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length/log_interval
            running_reward = running_reward/log_interval
            
            print('Episode {} \t Avg length: {:.2f} \t Avg reward: {:.2f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
    
    K, _, _ = control.dlqr(A,B,Q,R)
    
    x0 = np.random.randn(1,)
    u0 = np.zeros((1,))
    T = 50
    
    x_star, u_star = control.simulate_discrete(A,B,K,x0.reshape(1,1),u0.reshape(1,1),T)
    x_sim, u_sim = simulate(A,B,ppo.policy.actor,x0,u0,T)
    
    compare_paths(np.array(x_sim), np.squeeze(x_star[:,:-1]))
    compare_paths(np.array(u_sim), np.squeeze(u_star[:,:-1]))
    compare_V(ppo.policy.critic,A,B,Q,R,K,T,gamma,alpha)
    
    