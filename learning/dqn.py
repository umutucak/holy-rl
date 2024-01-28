"""Deep Q-Learning implementation from scratch.

Heavily inspired from cleanrl.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

class QNetwork(nn.Module):
    """DQN Network."""
    def __init__(self, env:gym.Env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, np.array(env.action_space.shape).prod())
        )

    def forward(self, x:np.ndarray):
        """Feed forward.
        
        Args:
        - `x (np.ndarray)`: Input observation of the network.
        """
        return self.network(x)
