import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

from _04_COP_POINTING_AND_ATTENTION._02_DQN_ATTN_MKP.e_qnet_attn import MultiHeadSelfAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorAttn(nn.Module):
    def __init__(
            self,
            n_features: int,  # 1 + n_resources
            hiddens: int = 32,
            n_heads: int = 1,
    ):
        super().__init__()
        n_resource = n_features - 1

        d_model = n_resource * 32 * n_heads

        # (batch_size, n_items, n_resource + 1) -> (batch_size, n_items, d_model)
        self.emd = nn.Linear(n_features, d_model)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, d_model)
        self.attn_layers = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, 1)
        self.linear_layers = nn.Sequential(
            nn.Linear(d_model, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, 1)
        )

        self.to(DEVICE)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(DEVICE)
        else:
            raise TypeError(f"unknown type: {type(x)}")

        # (n_items, 1+n_resource) -> (batch_size, n_items, 1+n_resource)
        no_batch = False
        if x.ndim == 2:
            no_batch = True
            x = x.unsqueeze(0)

        # (batch_size, n_items, 1 + n_resource) -> (batch_size, n_items, d_model)
        x = self.emd(x)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, d_model)
        x = x + self.attn_layers(x)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, 1)
        x = self.linear_layers(x)

        # (batch_size, n_items, 1) -> (batch_size, n_items)
        x = x.squeeze(-1)

        if no_batch:
            x = x.squeeze(0)

        mu_v = F.softmax(x, dim=-1)

        return mu_v

    def get_action(self, obs, action_mask, exploration: bool = True) -> np.ndarray:
        if isinstance(action_mask, list):
            action_mask = np.array(action_mask, dtype=np.float32)
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).bool().to(DEVICE)
        elif isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.bool().to(DEVICE)
        else:
            raise TypeError(f"unknown type: {type(action_mask)}")

        # obs.shape: (batch_size or n_envs, num_features) or (num_features,)
        # action_mask.shape: (batch_size or n_envs, num_items) or (num_items,)

        mu_v = self.forward(obs)
        mu_v = mu_v.masked_fill(action_mask, 0.0)

        if exploration:
            dist = Categorical(probs=mu_v)
            action = dist.sample()
            action = action.detach().numpy()
        else:
            action = torch.argmax(mu_v).detach().numpy()

        return action


class CriticAttn(nn.Module):
    """
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    """

    def __init__(
            self,
            n_features: int,  # 1 + n_resources
            n_items: int,
            hiddens: int = 32,
            n_heads: int = 1,
    ):
        super().__init__()
        n_resource = n_features - 1
        n_items = n_items

        d_model = n_resource * 32 * n_heads

        # (batch_size, n_items, n_resource + 1) -> (batch_size, n_items, d_model)
        self.emd = nn.Linear(n_features, d_model)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, d_model)
        self.attn_layers = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)

        # (batch_size, n_items * d_model) -> (batch_size, 1)
        self.linear_layers = nn.Sequential(
            nn.Linear(n_items * d_model, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, 1)
        )

        self.to(DEVICE)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(DEVICE)
        else:
            raise TypeError(f"unknown type: {type(x)}")

        # (n_items, 1+n_resource) -> (batch_size, n_items, 1+n_resource)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # (batch_size, n_items, 1 + n_resource) -> (batch_size, n_items, d_model)
        x = self.emd(x)

        # (batch_size, n_items, d_model) -> (batch_size, n_items, d_model)
        x = x + self.attn_layers(x)

        # (batch_size, n_items, d_model) -> (batch_size, n_items * d_model)
        x = torch.flatten(x, start_dim=1)

        # (batch_size, n_items * d_model) -> (batch_size, 1)
        x = self.linear_layers(x)

        return x


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done', 'action_mask']
)


class Buffer:
    def __init__(self):
        self.buffer = collections.deque()

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def get(self):
        observations, actions, next_observations, rewards, dones, action_masks = zip(*self.buffer)

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
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)
        action_masks = torch.tensor(action_masks, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones, action_masks
