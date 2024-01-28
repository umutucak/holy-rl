"""Deep Q-Learning implementation from scratch.

Heavily inspired from cleanrl.
"""

import random

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
    lr:float = 0.01
    buffer_size:int = 10000 # experience replay buffer size
    gamma: float = 0.99 # discount factor
    batch_sze: int = 128 # batch size for experience replay buffer sampling
    epsilon_max: float = 1 # starting epsilon value (exploration/exploitation)
    epsilon_min:float = 0.05 # ending epsilon value
    tnur: int = 1 # target network update rate
    tnuf: int = 1 # target network update frequency
    qntf: int = 10 # qnetwork training frequency

    # Initialize RNG seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create gym environment
    env = gym.make("CartPole-v1")

    # Utilize GPU for training if GPU present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent & target network
    q_net = QNetwork(env=env).to(device)
    target_net = QNetwork(env=env).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    erb = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        handle_timeout_termination=False
    )
