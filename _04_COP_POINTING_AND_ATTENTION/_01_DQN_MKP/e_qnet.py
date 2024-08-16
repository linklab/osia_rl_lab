import random
from torch import nn
import torch.nn.functional as F
import collections
import torch
import numpy as np


class QNet(nn.Module):
    def __init__(self, n_features, n_actions, device='cpu'):
        super(QNet, self).__init__()

        self.n_features = n_features
        self.n_actions = n_actions

        hiddens = 256

        self.layers = nn.Sequential(
            nn.Linear(n_features, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, n_actions)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if x.ndim == 3:
            x = torch.flatten(x, start_dim=1)

        x = self.layers(x)

        return x

    def get_action(self, obs, epsilon, action_mask):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if obs.ndim == 2:
            obs = torch.flatten(obs, start_dim=0)

        # random.random(): 0.0과 1.0사이의 임의의 값을 반환
        if random.random() < epsilon:
            available_actions = np.where(action_mask == 0.0)[0]
            action = random.choice(available_actions)
        else:
            q_values = self.forward(obs)
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(action_mask, -float('inf'))
            action = torch.argmax(q_values, dim=-1)
            action = action.item()

        return action  # argmax: 가장 큰 값에 대응되는 인덱스 반환


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done', 'action_mask']
)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def size(self):
        return len(self.buffer)

    def is_full(self):
        return self.size() >= self.capacity

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones, action_masks = zip(
            *[self.buffer[idx] for idx in indices]
        )

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        action_masks = np.array(action_masks)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

        return observations, actions, next_observations, rewards, dones, action_masks