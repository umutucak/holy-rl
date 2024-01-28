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

if __name__ == "__main__":
    # Parameters
    seed:int = 42 #rng seed
    total_timesteps:int = 500000 # timestep max of an experiment
    learning_rate:float = 0.01
    buffer_size:int = 10000 # experience replay buffer size
    gamma: float = 0.99 # discount factor
    batch_sze: int = 128 # batch size for experience replay buffer sampling
    epsilon_max: float = 1 # starting epsilon value (exploration/exploitation)
    epsilon_min:float = 0.05 # ending epsilon value
    tnur: int = 1 # target network update rate
    tnuf: int = 1 # target network update frequency
    qntf: int = 10 # qnetwork training frequency

    env = gym.make("CartPole-v1")
