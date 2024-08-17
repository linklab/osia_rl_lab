import os
import random
import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) must be divisible by num_heads({num_heads})"

        self.d_model = d_model  # dimension of model
        self.num_heads = num_heads  # number of heads
        self.d_k = d_model // num_heads  # dimension of each head

        self.w_q = nn.Linear(d_model, d_model)  # wieght for query
        self.w_k = nn.Linear(d_model, d_model)  # wieght for key
        self.w_v = nn.Linear(d_model, d_model)  # wieght for value
        self.w_o = nn.Linear(d_model, d_model)  # wieght for output

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        batch_size, seq_length, _ = x.size()

        # Compute query, key, value matrices
        # (batch_size, num_heads, seq_length, d_k)
        q = self.w_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate scaled dot-product attention
        # (batch_size, num_heads, seq_length, seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

        # Multiply attention probabilities with value matrix
        # (batch_size, num_heads, seq_length, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate and apply output linear layer
        # (batch_size, seq_length, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.w_o(attn_output)

        return output


class QNetAttn(nn.Module):
    def __init__(
            self,
            n_features: int,  # 1 + n_resources
            hiddens: int = 32,
            n_heads: int = 1,
            device='cpu'
    ):
        super(QNetAttn, self).__init__()

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

    def forward(self, x):
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

        return x

    def get_action(self, obs, epsilon, action_mask):
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
        if random.random() < epsilon:
            assert action_mask.ndim == 1
            available_actions = np.where(action_mask == 0.0)[0]
            actions = np.random.choice(available_actions)
        else:
            q_values = self.forward(obs)
            q_values = q_values.masked_fill(action_mask, -float('inf'))
            actions = torch.argmax(q_values, dim=-1)
            actions = actions.cpu().numpy()

        return actions  # argmax: 가장 큰 값에 대응되는 인덱스 반환


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done', 'action_mask']
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

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
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)
        action_masks = torch.tensor(action_masks, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones, action_masks


def attention_test():
    # 사용 예시:
    d_model = 128
    num_heads = 8
    batch_size = 32
    seq_length = 50

    x = torch.rand(batch_size, seq_length, d_model)  # 임의의 입력 텐서
    multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads)
    output = multi_head_self_attention(x)
    print(x.shape)
    print(output.shape)  # 출력: torch.Size([32, 50, 128])


if __name__ == '__main__':
    attention_test()