import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetCNN(nn.Module):
    def __init__(self, n_actions: int = 6):
        super().__init__()
        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.LazyLinear(self.n_actions),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            x = torch.unsqueeze(x, dim=0)
        x = self.network(x)
        return x

    def get_action(self, obs: torch.Tensor, epsilon: float = 0.1) -> int:
        # random.random(): 0.0과 1.0사이의 임의의 값을 반환
        if random.random() < epsilon:
            action = random.randrange(0, self.n_actions)
        else:
            q_values = self.forward(obs)
            action = torch.argmax(q_values, dim=-1)
            action = action.item()
        return action  # argmax: 가장 큰 값에 대응되는 인덱스 반환


Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
