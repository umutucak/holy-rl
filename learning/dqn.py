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
            nn.Linear(84, env.action_space.n)
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
    epsilon: float = 1 # starting epsilon value (exploration/exploitation)
    epsilon_min:float = 0.05 # ending epsilon value
    epsilon_decay:float = 0.001 # epsilon decay rate to go from max to min
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
    q_net = QNetwork(env).to(device)
    target_net = QNetwork(env=env).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    # Initialize experience replay buffer
    erb = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        handle_timeout_termination=False
    )

    # gym env game loop start
    obs, infos = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        # getting an action using epsilon
        if random.random() < epsilon: # exploration
            action = env.action_space.sample() # random action
        else: #exploitation
            q_values = q_net(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values).cpu().numpy() # action with highest q_value
        epsilon = max(epsilon-epsilon_decay, epsilon_min) # decay the epsilon

        # Step through the environment to get obs and reward
        next_obs, reward, term, trunc, infos = env.step(action)

        # Enter data into experience replay buffer
        erb.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=term,
            infos=infos
        )

        # update obs for next iter
        obs = next_obs