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


def compare_P(actor, K, low=-5, high=5, actor_label='Approx. Policy'):
    """
    actor is expected to be a pytorch nn.Sequential model
    """
    fig, ax = plt.subplots()
    colrs = ['#2D328F', '#F15C19']  # blue, orange
    label_fontsize = 18

    states = torch.linspace(low, high, 1000).detach().reshape(1000, 1)
    actions = actor(states).squeeze().detach().numpy()
    optimal = -K * states.numpy()

    ax.plot(states.numpy(), optimal, color=colrs[1], label='Optimal Policy')
    ax.plot(states.numpy(), actions, color=colrs[0], label=actor_label)

    ax.set_xlabel('x (state)', fontsize=label_fontsize)
    ax.set_ylabel('u (action)', fontsize=label_fontsize)
    plt.legend()

    plt.grid(True)
    plt.show()
    return

def plot_cost(A,B,Q,R,gamma=1, size="small"):
    """

    Parameters
    ----------
    A,B,Q,R : 2d list
        [[r]] for real number r

    gamma : float, optional
        discount factor, default 1.
    size : string, optional
        accepted values are "small" and "large"
        indicates type of graph, either zoomed in or not

    """
    def _gen_cost(t0, t1, gamma):
        with torch.no_grad():
            policy = FourTents(t0,t1)
            
            running_cost = 0
            state = torch.tensor([[-117/59]])   
            for i in range(10):        
                action = policy(state) 
                
                cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action
                
                state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action

                running_cost += gamma**i * cost.item()
                
            # print("x_0 done")
                
            state = torch.tensor([[-588/295]])
            for i in range(10):
                action = policy(state) 
                
                cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action
                
                state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action

                running_cost += gamma**i * cost.item()
            
        return running_cost
    

    
    if size == "small":
        x_low = -0.001
        x_high = 0.001
        y_low = 0.999
        y_high = 1.001
    elif size == "large":
        x_low = -0.01
        x_high = 0.01
        y_low = 0.99
        y_high = 1.01
    else:
        raise NotImplementedError()
    
    t0_range = np.linspace(x_low, x_high, 100)
    t1_range = np.linspace(y_low, y_high, 100)
    
    X, Y = np.meshgrid(t0_range, t1_range)
    costs = np.array([[_gen_cost(t0, t1, gamma) for t0 in t0_range] for t1 in t1_range])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.contour3D(X, Y, costs, 500, cmap="coolwarm")
    ax1.set_xlabel(r"$\theta_0$")
    ax1.set_ylabel(r"$\theta_1$")
    ax1.set_zlabel("Cost")
    ax1.set_xticks([x_low, 0, x_high])
    ax1.set_yticks([y_low, 1.00, y_high])
    ax1.set_zticks([14.0, 15.0])
    ax1.set_title("Cost Landscape")
    ax1.xaxis.set_rotate_label(False) 
    ax1.yaxis.set_rotate_label(False) 
    ax1.view_init(elev=17, azim=-108)
    
    ax2.contour3D(X, Y, costs, 500, cmap="coolwarm")
    ax2.set_xlabel(r"$\theta_0$")
    ax2.set_ylabel(r"$\theta_1$")
    ax2.set_zlabel("Cost")
    ax2.set_xticks([x_low, 0, x_high])
    ax2.set_yticks([y_low, 1.00, y_high])
    ax2.set_zticks([14.0, 15.0])
    ax2.set_title("Cost Landscape")
    ax2.xaxis.set_rotate_label(False) 
    ax2.yaxis.set_rotate_label(False) 
    ax2.view_init(elev=71, azim=-128)
    return


# "custom" activation functions for pytorch - compatible with autograd
class PLU(nn.Module):
    def __init__(self):
        super(PLU, self).__init__()
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.w2 = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.max(x, torch.zeros_like(x)) + self.w2 * torch.min(x, torch.zeros_like(x))

class Tent(nn.Module):
    def __init__(self, height=1, center=0, width=1):
        super(Tent, self).__init__()
        self.c = center
        self.w = width
        self.h = height

    def forward(self, x):
        return self.h * torch.max(torch.min((self.w + self.c - x)/self.w, (x - (self.c - self.w))/self.w ), torch.zeros_like(x))


class LinSpike(nn.Module):
    def __init__(self, height=2, center=1, width=1):
        super(LinSpike, self).__init__()
        self.tent = Tent(height, center, width)
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.alpha * x + self.beta * self.tent(x)


class FourTents(nn.Module):
    def __init__(self, alpha, beta):
        super(FourTents, self).__init__()
        self.alpha = torch.nn.Parameter(torch.FloatTensor([alpha]))
        self.beta = torch.nn.Parameter(torch.FloatTensor([beta]))
        
        self.w_0, self.w_1, self.w_2, self.w_3 = (-0.5, 3, 0.5, 0.5)
        self.a_1, self.a_2, self.a_3 = (-2, 1.5, 1.8)
        self.d_1, self.d_2, self.d_3 = (0.1, 0.0005, 0.0005)
        
        self.tent_1 = Tent(center=self.a_1, width=self.d_1)
        self.tent_2 = Tent(height=self.w_2, center=self.a_2, width=self.d_2)
        self.tent_3 = Tent(height=self.w_3, center=self.a_3, width=self.d_3)
        
        
    def forward(self, x):
        return self.alpha * x + self.beta * (self.w_0 * torch.abs(x) + self.w_1 * self.tent_1(x) + self.tent_2(x) + self.tent_3(x))


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
            LinSpike(),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action = self.agent(state)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)

        return action

class CONTEXAMPLE(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, alpha, beta):
        super(CONTEXAMPLE, self).__init__()

        self.agent = nn.Sequential(
            FourTents(alpha, beta),
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
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, alpha=0, beta=1):
        self.betas = betas
        self.gamma = gamma

        ## uncomment/comment to switch between activation function variants
        # self.policy = PRELU(state_dim, action_dim, n_latent_var).to(device)
        # self.policy = CHAOS(state_dim, action_dim, n_latent_var).to(device)
        # self.policy = LINEAR(state_dim, action_dim, n_latent_var).to(device)
        self.policy = CONTEXAMPLE(state_dim, action_dim, n_latent_var, alpha, beta).to(device)

        self.optimizer = torch.optim.SGD(self.policy.agent.parameters(), lr=lr)

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
        return costs

if __name__ == "__main__":
    
    ############## Hyperparameters ##############
    
    A = np.array(0).reshape(1, 1)
    B = np.array(1).reshape(1, 1)
    Q = np.array(1).reshape(1, 1)
    R = np.array(0).reshape(1, 1)
    
    state_dim = 1
    action_dim = 1
    log_interval = 500  # print avg cost in the interval
    max_episodes = 5000000  # max training episodes
    max_timesteps = 10  # max timesteps in one episode
    
    n_latent_var = 1  # number of variables in hidden layer
    
    gamma = 0.99  # discount factor
    sigma = 0.1  # std dev of exploratory policy
    
    lr = 5e-7
    betas = (0.9, 0.999)  # parameters for Adam optimizer
    
    random_seed = 1
    
    state_tol = 0.01
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
    print(f"device: {device}, lr: {lr}, betas: {betas}")
    
    # logging variables
    running_cost = 0
    
    # pre-images
    x_0 = -117/59
    y_0 = -588/295
    
    # training loop
    for i_episode in range(1, max_episodes + 1):
        # state = torch.normal(0, 1, size=(1, 1))
        
        # if np.random.uniform(0, 1) < 2.78e-10:
        #     state = torch.uniform(-5,5, size=(1,1))
        # else:
        # if np.random.uniform(0,1) < 0.5:
        #     state = torch.tensor([[x_0]])
        # else:
        #     state = torch.tensor([[y_0]])
        
        state = torch.tensor([[x_0]])   
        done = False
        while not done:
            action = pg.select_action(state, memory) 
    
            cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action
    
            state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
            done = (-state_tol < state < state_tol)  # if state is basically zero
        
            # Saving cost and is_terminals:
            memory.costs.append(cost)
            memory.is_terminals.append(done)
    
            running_cost += cost.item()
            
        state = torch.tensor([[y_0]])
        done = False
        while not done:
            action = pg.select_action(state, memory) 
    
            cost = state * torch.FloatTensor(Q) * state + action * torch.FloatTensor(R) * action
    
            state = torch.FloatTensor(A) * state + torch.FloatTensor(B) * action
            done = (-state_tol < state < state_tol)  # if state is basically zero
        
            # Saving cost and is_terminals:
            memory.costs.append(cost)
            memory.is_terminals.append(done)
    
            running_cost += cost.item()
    
        if i_episode % 1 == 0:
            pg.update(memory)
            memory.clear_memory()
    
        # logging
        if i_episode % log_interval == 0:
            print('Episode {} \t Avg cost: {:.2f}'.format(i_episode, running_cost / log_interval))
            # print('\t Distance to optimal parameters: {:.2f}'.format(torch.norm(torch.tensor(list(pg.policy.agent.parameters()))-optimal_params), 2))
            print(list(pg.policy.agent.parameters()))
            running_cost = 0
    
    print(list(pg.policy.agent.parameters()))
    
    # random init to compare how the two controls act
    x0 = np.random.randn(1, 1)
    T = 50
    
    x_star, u_star = control.simulate_discrete(A, B, K, x0.reshape(1, 1), T)
    x_sim, u_sim = simulate(A, B, pg.policy.agent, x0, T)
    
    compare_paths(np.array(x_sim), np.squeeze(x_star[:, :-1]), "state")
    compare_paths(np.array(u_sim), np.squeeze(u_star[:, :-1]), "action")
    
    compare_P(pg.policy.agent, K)
