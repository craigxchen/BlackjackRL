import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from gridworld_environment import GridWorld

log_interval = 100
homotopy = False

max_episodes = 100000
if homotopy:
    gamma = 0.01
else:
    gamma = 1.00
lr = 1e-3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = GridWorld()

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Softmax(),
).to(device)

adam_opt = optim.Adam(model.parameters(), lr=lr)

# set to training mode
model.train()

# logging info
running_reward = 0
exp_length = 0
loss_tracker = []

for idx in range(max_episodes):
    state, _, done = env.reset()
    state = torch.tensor([state]).float().to(device)
    rewards = []
    log_probs = []

    while not done:
        # select action
        action_probs = model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # take next step in environment
        next_state, reward, done = env.step(action.numpy())
        state = torch.tensor([next_state]).float().to(device)

        # store relevant info
        rewards.append(reward)
        log_probs.append(action_logprob.unsqueeze(-1))

        # logging
        running_reward += reward
        exp_length += 1

    # calculated discounted rewards
    discounted_rewards = []
    dr = 0
    for r in reversed(rewards):
        dr = r + gamma * dr
        discounted_rewards.insert(0, dr)

    # normalize
    rewards = torch.tensor(discounted_rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    loss = []
    for log_prob, rew in zip(log_probs, rewards):
        loss.append(-log_prob * rew)

    # take gradient ascent step
    adam_opt.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    adam_opt.step()

    loss_tracker.append(loss.item())
    loss_tracker = loss_tracker[-11:]  # only keep previous 10 items (excluding the current loss)
    if len(loss_tracker) > 10 and gamma < 1:
        # if loss hasn't changed in the past 10 iterations, we are probably near some minima, increase gamma
        if abs(sum(loss_tracker[:-1])/10 - loss_tracker[-1]) < 0.01:
            gamma += 0.01
            print(f"Gamma increased to {gamma:.2f}!")
            del loss_tracker[:]

    if idx % log_interval == 0 and idx != 0:
        if running_reward / log_interval >= -50.0:
            print(f"episode: {idx} \t avg length: {exp_length / log_interval} \t avg reward: {running_reward / log_interval}")
            print("Solved!")
            break
        print(f"episode: {idx} \t avg length: {exp_length / log_interval} \t avg reward: {running_reward / log_interval}")
        running_reward = 0
        exp_length = 0



