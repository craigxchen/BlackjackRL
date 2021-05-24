import torch
import numpy as np
# import lqr_control as control
from DeterministicPG_1dim import PG, Memory

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
max_episodes = 200000  # max training episodes

n_latent_var = 1  # number of variables in hidden layer

gamma = 0.01  # discount factor
gamma_lr = 0.01  # update step size for discount factor

lr = 1e-3

# NOT APPLICABLE; WE ARE USING SGD
betas = (0.9, 0.999)  # parameters for Adam optimizer

random_seed = 1
state_tol = 1e-2  # tolerance for terminal states
param_tol = 1e-3  # tolerance for gamma updating
#############################################

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
pg = PG(state_dim, action_dim, n_latent_var, lr, betas, gamma)

# Optimal control for comparison
# K, P, _ = control.dlqr(A, B, Q, R)
# optimal_params = torch.FloatTensor([-K.item(),0.])

# compare initial policy with optimal
# compare_P(pg.policy.agent, K, actor_label="Initial Policy")

# important parameters
print(f"device: {device}, lr: {lr}, gamma: {gamma}")

# logging variables
running_cost = 0
cost_tracker = []  # cost per episode to plot
gamma_update_tracker = []  # track i_episodes when gamma was updated


# pre-images
x_0 = -117/59
y_0 = -588/295

# training loop
for i_episode in range(1, max_episodes + 1):    
    
    state = torch.tensor([[x_0]])   
    x_cost = 0
    t = 0
    done = False
    while not done:
        action = pg.select_action(state, memory) 

        cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action

        state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
        done = (-state_tol < state.item() < state_tol)  # if state is basically zero
    
        # Saving cost and is_terminals:
        memory.costs.append(cost)
        memory.is_terminals.append(done)

        running_cost += cost.item()
        x_cost += (gamma**t) * cost.detach().item()  # since 0**0 = 1 in python
        t += 1
    
    state = torch.tensor([[y_0]])
    y_cost = 0
    t = 0
    done = False
    while not done:
        action = pg.select_action(state, memory) 

        cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action

        state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
        done = (-state_tol < state.item() < state_tol)  # if state is basically zero
    
        # Saving cost and is_terminals:
        memory.costs.append(cost)
        memory.is_terminals.append(done)

        running_cost += cost.item()
        y_cost += (gamma**t) * cost.detach().item()  # since 0**0 = 1 in python
        t += 1

    # update parameters
    pg.update(memory)
    memory.clear_memory()
    
    # tracking parameter evolution via gradient
    # update gamma if gradient is essentially 0
    if (-param_tol < pg.policy.agent[0].alpha.grad.item() < param_tol) and (-param_tol < pg.policy.agent[0].beta.grad.item() < param_tol) and gamma < 1:
        gamma += gamma_lr
        pg.gamma = gamma
        print(f"Gamma increased to {gamma:.2f}!")
        gamma_update_tracker.append(i_episode)

    # for plotting cost
    cost_tracker.append(x_cost + y_cost)

    # logging
    if i_episode % log_interval == 0:
        print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost / log_interval))
        # print('\t Distance to optimal parameters: {:.2f}'.format(torch.norm(torch.tensor(list(pg.policy.agent.parameters()))-optimal_params), 2))
        print(list(pg.policy.agent.parameters()))
        print(f"alpha grad: {pg.policy.agent[0].alpha.grad.item()} \t beta grad: {pg.policy.agent[0].beta.grad.item()}")
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

fig = plt.figure()
ax = fig.add_subplot(111)
t = np.array(list(range(max_episodes)))
ax.plot(t, np.array(cost_tracker[150000:]), 'b-')
ax.scatter(np.array(gamma_update_tracker), np.array([cost_tracker[x] for x in gamma_update_tracker]), color="#F15C19")
ax.set_xscale('linear')
