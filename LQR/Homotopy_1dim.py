import torch
import numpy as np
import lqr_control as control
from DeterministicPG_1dim import PG, Memory

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

############## Hyperparameters ##############

A = np.array(0.1).reshape(1, 1)
B = np.array(1).reshape(1, 1)
Q = np.array(1).reshape(1, 1)
R = np.array(0.1).reshape(1, 1)

state_dim = 1
action_dim = 1
log_interval = 500  # print avg cost in the interval
max_episodes = 500000  # max training episodes
max_timesteps = 5  # max timesteps in one episode

n_latent_var = 1  # number of variables in hidden layer

gamma = 0.01  # discount factor
gamma_lr = 0.01  # update step size for discount factor
lr = 1e-3

# NOT APPLICABLE; WE ARE USING SGD
betas = (0.9, 0.999)  # parameters for Adam optimizer

random_seed = 1
state_tol = 1e-2  # tolerance for terminal states
param_tol = 1e-4  # tolerance for gamma updating
#############################################

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
pg = PG(state_dim, action_dim, n_latent_var, lr, betas, gamma)

# to manually edit parameters
state_dict = pg.policy.state_dict()
state_dict['agent.0.alpha'] = torch.tensor([0.0])
state_dict['agent.0.beta'] = torch.tensor([0.0])
pg.policy.load_state_dict(state_dict)


# important parameters
print(f"device: {device}, lr: {lr}, gamma: {gamma}")

# logging variables
running_cost = 0
cost_tracker = [] # cost per episode to plot
gamma_update_tracker = [0]  # track i_episodes when gamma was updated


# pre-images
x_0 = -117/59
y_0 = -588/295

# training loop
for i_episode in range(1, max_episodes + 1):    
    
    state = torch.tensor([[x_0]])   
    x_cost = 0
    # t = 0
    done = False
    for i in range(max_timesteps):
        action = pg.select_action(state, memory) 

        cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action

        state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
        # done = (-state_tol < state.item() < state_tol)  # if state is basically zero
        done = (i == max_timesteps - 1)
    
        # Saving cost and is_terminals:
        # memory.costs.append(cost)
        # memory.is_terminals.append(done)

        running_cost += cost.item()
        x_cost += (gamma**i) * cost  # since 0**0 = 1 in python
        # t += 1
        
    memory.costs.append(x_cost)
    memory.is_terminals.append(True)
    
    state = torch.tensor([[y_0]])
    y_cost = 0
    # t = 0
    done = False
    for i in range(max_timesteps):
        action = pg.select_action(state, memory) 

        cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action

        state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
        # done = (-state_tol < state.item() < state_tol)  # if state is basically zero
        done = (i == max_timesteps - 1)
    
        # Saving cost and is_terminals:
        # memory.costs.append(cost)
        # memory.is_terminals.append(done)

        running_cost += cost.item()
        y_cost += (gamma**i) * cost  # since 0**0 = 1 in python
        # t += 1

    memory.costs.append(y_cost)
    memory.is_terminals.append(True)

    # update parameters
    pg.update(memory)
    memory.clear_memory()
    
    # tracking parameter evolution via gradient
    # update gamma if gradient is essentially 0
    if (-param_tol < pg.policy.agent[0].alpha.grad.item() < param_tol) and (-param_tol < pg.policy.agent[0].beta.grad.item() < param_tol) and gamma < 1 and (i_episode - gamma_update_tracker[-1]) >= 500:
        gamma += gamma_lr
        pg.gamma = gamma
        print(f"Gamma increased to {gamma:.2f}!")
        gamma_update_tracker.append(i_episode)

    # for plotting cost
    cost_tracker.append((x_cost + y_cost).item())

    # logging
    if i_episode % log_interval == 0:
        print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost / log_interval))
        # print('\t Distance to optimal parameters: {:.2f}'.format(torch.norm(torch.tensor(list(pg.policy.agent.parameters()))-optimal_params), 2))
        print(list(pg.policy.agent.parameters()))
        print(f"alpha grad: {pg.policy.agent[0].alpha.grad.item()} \t beta grad: {pg.policy.agent[0].beta.grad.item()} \n")
        running_cost = 0

print(list(pg.policy.agent.parameters()))

# random init to compare how the two controls act
# x0 = np.random.randn(1, 1)
# T = 50

# x_star, u_star = control.simulate_discrete(A, B, K, x0.reshape(1, 1), T)
# x_sim, u_sim = simulate(A, B, pg.policy.agent, x0, T)

# compare_paths(np.array(x_sim), np.squeeze(x_star[:, :-1]), "state")
# compare_paths(np.array(u_sim), np.squeeze(u_star[:, :-1]), "action")

# compare_P(pg.policy.agent, K)

# %% plotting cost over training
import matplotlib.pyplot as plt

def opt_cost(A,B,Q,R,gamma=1):
    x_0 = -117/59
    y_0 = -588/295

    # Optimal control for comparison
    K, P, _ = control.dlqr(np.sqrt(gamma) * A, B, Q, R/gamma)
    
    # state = torch.tensor([[x_0]])   
    # x_cost = 0

    # for i in range(max_timesteps):
    #     action = torch.FloatTensor([[-K.item()]]) * state
        
    #     cost = gamma**i * (state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action)
       
    #     x_cost += cost.item()  # since 0**0 = 1 in python
        
    #     state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action

    
    x_cost = P.item() * x_0 **2
    
    # state = torch.tensor([[y_0]])   
    # y_cost = 0

    # for i in range(max_timesteps):
    #     action = torch.FloatTensor([[-K.item()]]) * state

    #     cost = gamma ** i * (state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action)
        
    #     y_cost += cost.item()  # since 0**0 = 1 in python
        
    #     state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
        
    y_cost = P.item() * y_0 **2
    
    return x_cost + y_cost

zoom_low = 0 #gamma_update_tracker[1]
zoom_high = 10000 #gamma_update_tracker[-1]

fig = plt.figure()
ax1 = fig.add_subplot(121)
t = np.array(list(range(zoom_low,zoom_high)))
ax1.plot(t, np.array(cost_tracker[zoom_low:zoom_high]), 'b-', zorder=1)

for x in range(len(gamma_update_tracker)-1):
    if zoom_low <= gamma_update_tracker[x] <= zoom_high:
        ax1.hlines(opt_cost(A,B,Q,R,gamma=(1+x) * gamma_lr), gamma_update_tracker[x], gamma_update_tracker[x+1], colors="r", zorder=3)
        if x == len(gamma_update_tracker)-2:
            ax1.hlines(opt_cost(A,B,Q,R), gamma_update_tracker[x+1], zoom_high, colors="r")

ax1.scatter(np.array([x for x in gamma_update_tracker[1:] if zoom_low <= x <= zoom_high]), np.array([cost_tracker[x] for x in gamma_update_tracker[1:] if zoom_low <= x <= zoom_high]), color="#F15C19", zorder=2)
ax1.set_xscale('linear')
ax1.set_xlabel(r"Training Iterations", fontsize=24)
ax1.set_ylabel(r"Cost", fontsize=24)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2 = fig.add_subplot(122)
t = np.array(list(range(max_episodes)))
ax2.plot(t, np.array(cost_tracker), 'b-')

# for x in range(len(gamma_update_tracker)-1):
#     ax2.hlines(opt_cost(A,B,Q,R,gamma=(1+x) * gamma_lr), gamma_update_tracker[x], gamma_update_tracker[x+1], colors="r", zorder=3)
ax2.hlines(opt_cost(A,B,Q,R), 0, max_episodes, colors="r", linestyles="dashed")

ax2.scatter(np.array(gamma_update_tracker[1:]), np.array([cost_tracker[x-1] for x in gamma_update_tracker[1:]]), color="#F15C19")
ax2.set_xscale('linear')
ax2.set_xlabel(r"Training Iterations", fontsize=24)
ax2.set_ylabel(r"Cost", fontsize=24)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)