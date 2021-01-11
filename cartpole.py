import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt

log_interval = 100

max_episodes = 100000
gamma = 0.01
lr = 1e-3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = gym.make("CartPole-v1")
render = False
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n

model = nn.Sequential(
    nn.Linear(state_space_dim, 64),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(64, action_space_dim),
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
    state = torch.from_numpy(env.reset()).float().to(device)
    rewards = []
    log_probs = []
    done = False

    while not done:
        # select action
        action_probs = model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # take next step in environment
        next_state_numpy, reward, done, _ = env.step(action.numpy())
        if render:
            env.render()
        state = torch.from_numpy(next_state_numpy).float().to(device)

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

    if idx % log_interval == 0:
        print(f"episode: {idx} \t avg length: {exp_length / log_interval} \t avg reward: {running_reward / log_interval}")
        if running_reward / log_interval > env.spec.reward_threshold:
            print("Solved! Reward above threshold")
            break
        running_reward = 0
        exp_length = 0



