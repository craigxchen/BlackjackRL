# temp fix for OpenMP issue
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import lqr_control as control

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def simulate(A, B, policy, x0, T):
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
        x = A @ x + B @ u.numpy()

    return x_data, u_data


def compare_paths(x_sim, x_star, ylabel):
    fig, ax = plt.subplots()
    colors = ['#2D328F', '#F15C19']  # blue, orange

    t = np.arange(0, x_star.shape[0])
    ax.plot(t, x_star, color=colors[1], label='True')
    ax.plot(t, x_sim, color=colors[0], label='Agent')

    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=18)

    plt.grid(True)
    plt.show()
    return


def compare_P(actor, K, low=-10, high=10, actor_label='Approx. Policy'):
    fig, ax = plt.subplots()
    colors = ['#2D328F', '#F15C19']  # blue, orange
    label_fontsize = 18

    states = torch.linspace(low, high).detach().reshape(100, 1)
    actions = actor(states).squeeze().detach().numpy()
    optimal = -K * states.numpy()

    ax.plot(states.numpy(), optimal, color=colors[1], label='Optimal Policy')
    ax.plot(states.numpy(), actions, color=colors[0], label=actor_label)

    ax.set_xlabel('x (state)', fontsize=label_fontsize)
    ax.set_ylabel('u (action)', fontsize=label_fontsize)
    plt.legend()

    plt.grid(True)
    plt.show()
    return


# "custom" activation functions for pytorch - compatible with autograd
class PLU(nn.Module):
    def __init__(self):
        super(PLU, self).__init__()
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.w2 = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.max(x, torch.zeros_like(x)) + self.w2 * torch.min(x, torch.zeros_like(x))


class Spike(nn.Module):
    def __init__(self, center=1, width=1):
        super(Spike, self).__init__()
        self.c = center
        self.w = width
        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.alpha * (
                torch.min(torch.max((x - (self.c - self.w)), torch.zeros_like(x)),
                          torch.max((-x + (self.c + self.w)), torch.zeros_like(x)))
                - 2 * torch.min(torch.max((x - (self.c - self.w + 1)), torch.zeros_like(x)),
                                torch.max((-x + (self.c + self.w + 1)), torch.zeros_like(x)))
        )


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


class LINEAR(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(LINEAR, self).__init__()

        self.agent = nn.Sequential(
            nn.Linear(state_dim, action_dim, bias=True)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action = self.agent(state)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)

        return action


class PRELU(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(PRELU, self).__init__()

        self.agent = nn.Sequential(
            PLU(),
            nn.Linear(state_dim, action_dim, bias=True)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action = self.agent(state)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)

        return action


class CHAOS(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(CHAOS, self).__init__()

        self.agent = nn.Sequential(
            nn.Linear(state_dim, action_dim, bias=True),
            Spike(),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action = self.agent(state)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)

        return action


class PG:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.betas = betas
        self.gamma = gamma

        ## uncomment/comment to switch between activation function variants
        self.policy = PRELU(state_dim, action_dim, n_latent_var).to(device)
        # self.policy = CHAOS(state_dim, action_dim, n_latent_var).to(device)
        # self.policy = LINEAR(state_dim, action_dim, n_latent_var).to(device)

        self.optimizer = torch.optim.Adam(self.policy.agent.parameters(), lr=lr, betas=betas)

    def select_action(self, state, memory):
        return self.policy.act(state, memory)

    def update(self, memory):
        # Monte Carlo estimate of state costs:
        costs = []
        discounted_cost = torch.zeros(1)
        for cost, is_terminal in zip(reversed(memory.costs), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_cost = torch.zeros(1)
            discounted_cost = cost + (self.gamma * discounted_cost)
            costs.insert(0, discounted_cost)

        costs = torch.stack(costs)

        self.optimizer.zero_grad()
        costs.mean().backward()
        self.optimizer.step()


############## Hyperparameters ##############

A = np.array(1).reshape(1, 1)
B = np.array(1).reshape(1, 1)
Q = np.array(1).reshape(1, 1)
R = np.array(1).reshape(1, 1)

state_dim = 1
action_dim = 1
log_interval = 100  # print avg cost in the interval
max_episodes = 200000  # max training episodes
max_timesteps = 10  # max timesteps in one episode

n_latent_var = 1  # number of variables in hidden layer

gamma = 0.99  # discount factor
sigma = 0.1  # std dev of exploratory policy

lr = 0.001
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

# compare initial policy with optimal
compare_P(pg.policy.agent, K, actor_label="Initial Policy")

# important parameters
print(f"device: {device}, lr: {lr}, betas: {betas}")

# logging variables
running_cost = 0

# training loop
for i_episode in range(1, max_episodes + 1):
    state = torch.normal(0, 1, size=(1, 1))
    done = False
    for t in range(max_timesteps):
        # Running exploratory policy
        action = pg.select_action(state, memory) + torch.normal(0, sigma, size=(1, 1))

        cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action

        if np.random.uniform(0, 1) > gamma:
            state = torch.normal(0, 1, size=(1, 1))
        else:
            state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action

        # Saving cost and is_terminals:
        memory.costs.append(cost)
        memory.is_terminals.append(done)

        if done:
            break

    pg.update(memory)

    memory.clear_memory()

    running_cost += cost.item()

    # logging
    if i_episode % log_interval == 0:
        print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost / log_interval))
        running_cost = 0

# random init to compare how the two controls act
x0 = np.random.randn(1, 1)
T = 50

x_star, u_star = control.simulate_discrete(A, B, K, x0.reshape(1, 1), T)
x_sim, u_sim = simulate(A, B, pg.policy.agent, x0, T)

compare_paths(np.array(x_sim), np.squeeze(x_star[:, :-1]), "state")
compare_paths(np.array(u_sim), np.squeeze(u_star[:, :-1]), "action")

compare_P(pg.policy.agent, K)
