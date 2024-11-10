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


class QNet(nn.Module):
    def __init__(self, n_actions: int = 2):
        super().__init__()
        self.n_actions = n_actions
        self.fc1 = nn.LazyLinear(out_features=512)  # fully connected
        self.fc2 = nn.LazyLinear(out_features=512)
        self.fc3 = nn.LazyLinear(out_features=512)
        self.fc4 = nn.LazyLinear(out_features=n_actions)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
