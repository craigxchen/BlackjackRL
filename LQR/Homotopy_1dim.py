import torch
import numpy as np
import lqr_control as control
from DeterministicPG_1dim import PG, Memory, compare_P, compare_paths, simulate

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

############## Hyperparameters ##############

A = np.array(0).reshape(1, 1)
B = np.array(1).reshape(1, 1)
Q = np.array(1).reshape(1, 1)
R = np.array(0).reshape(1, 1)

state_dim = 1
action_dim = 1
log_interval = 500  # print avg cost in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 10  # max timesteps in one episode

n_latent_var = 1  # number of variables in hidden layer

gamma = 0.00  # discount factor
gamma_lr = 0.01  # update step size for discount factor

lr = 1e-5
betas = (0.9, 0.999)  # parameters for Adam optimizer

random_seed = 1
#############################################

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
pg = PG(state_dim, action_dim, n_latent_var, lr, betas, gamma)

# Optimal control for comparison
K, P, _ = control.dlqr(A, B, Q, R)
optimal_params = torch.FloatTensor([-K.item(),0.])

# compare initial policy with optimal
compare_P(pg.policy.agent, K, actor_label="Initial Policy")

# important parameters
print(f"device: {device}, lr: {lr}, betas: {betas}")

# logging variables
running_cost = 0

# pre-images
x_0 = -1.9837
y_0 = -1.9935

for h_episode in range(1, int(1/gamma_lr)):
    # training loop
    for i_episode in range(1, max_episodes + 1):
        # state = torch.normal(0, 1, size=(1, 1))
        
        # if np.random.uniform(0, 1) < 2.78e-10:
        #     state = torch.uniform(-5,5, size=(1,1))
        # else:
        if np.random.uniform(0,1) < 0.5:
            state = torch.tensor([[x_0]])
        else:
            state = torch.tensor([[y_0]])
        
        done = False
        for t in range(max_timesteps):
            # Running exploratory policy
            action = pg.select_action(state, memory) # + torch.normal(0, sigma, size=(1, 1))
    
            cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action
    
            # if np.random.uniform(0, 1) > gamma:
            #     state = torch.normal(0, 1, size=(1, 1))
            # else:
            #     state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
    
            state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
    
            # Saving cost and is_terminals:
            memory.costs.append(cost)
            memory.is_terminals.append(done)
    
            running_cost += cost.item()
            
            if done:
                break
    
        if i_episode % log_interval == 0:
            pg.update(memory)
            memory.clear_memory()
    
        # logging
        if i_episode % log_interval == 0:
            print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost / log_interval))
            # print('\t Distance to optimal parameters: {:.2f}'.format(torch.norm(torch.tensor(list(pg.policy.agent.parameters()))-optimal_params), 2))
            print(list(pg.policy.agent.parameters()))
            running_cost = 0
            
    gamma += gamma_lr
    memory.clear_memory()

print(list(pg.policy.agent.parameters()))

# random init to compare how the two controls act
x0 = np.random.randn(1, 1)
T = 50

x_star, u_star = control.simulate_discrete(A, B, K, x0.reshape(1, 1), T)
x_sim, u_sim = simulate(A, B, pg.policy.agent, x0, T)

compare_paths(np.array(x_sim), np.squeeze(x_star[:, :-1]), "state")
compare_paths(np.array(u_sim), np.squeeze(u_star[:, :-1]), "action")

compare_P(pg.policy.agent, K)