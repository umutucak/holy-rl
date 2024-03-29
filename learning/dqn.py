"""Deep Q-Learning implementation from scratch.

Heavily inspired from cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py.
Studied from huggingface: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm.
"""

import random

import gymnasium as gym
import numpy as np
from tqdm import tqdm
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
    
def epsilon_decay(epsilon_max: float, epsilon_min: float, total_timesteps: int, current_timestep: int):
    slope = (epsilon_min - epsilon_max) / total_timesteps
    return max(slope * current_timestep + epsilon_max, epsilon_min)

if __name__ == "__main__":
    # Parameters
    seed:int = 42 #rng seed
    total_timesteps:int = 500000 # timestep max of an experiment
    lr:float = 2.5e-4
    buffer_size:int = 10000 # experience replay buffer size
    gamma: float = 0.99 # discount factor
    batch_size: int = 128 # batch size for experience replay buffer sampling
    epsilon_max: float = 1 # starting epsilon value (exploration/exploitation)
    epsilon_min:float = 0.05 # ending epsilon value
    epsilon_duration:float = 0.5 # time spent before min epsilon is reached
    training_start:int = 10000 # steps needed before training begins
    tnur: int = 1 # target network update rate
    tnuf: int = 500 # target network update frequency
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
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    # Target network is used to evaluate the progress of our DQN.
    # It represents the past policy from which we evaluate surplus reward gains.
    target_net = QNetwork(env=env).to(device)
    target_net.load_state_dict(q_net.state_dict())

    # Initialize Experience Replay (ER) buffer
    # ER is used in DQN to avoid catastrophic forgetting.
    # It allows the model to re-train on previous experiences in order to
    # mix it with novel experiences and not forget previous training.
    # Another benefit of ER is that by randomly sampling data from memory 
    # we avoid sequential correlation of experiences.
    erb = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        handle_timeout_termination=False
    )

    # Training begins
    obs, info = env.reset(seed=seed)
    done = False
    for global_step in tqdm(range(total_timesteps)):
        # getting an action using epsilon
        epsilon = epsilon_decay(epsilon_max, epsilon_min, epsilon_duration * total_timesteps, global_step)
        if random.random() < epsilon: # exploration
            action = env.action_space.sample() # random action
        else: #exploitation
            q_values = q_net(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values).cpu().numpy() # action with highest q_value
        # Step through the environment to get obs and reward
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        # Enter data into experience replay buffer
        erb.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=term,
            infos=info
        )
        # update obs for next iter
        obs = next_obs
        # Training
        if global_step > training_start:
            # Agent
            if global_step % qntf == 0:
                data = erb.sample(batch_size)
                with torch.no_grad():
                    # computing the TD Target
                    target_max, _ = target_net(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
                # computing current q_values
                value = q_net(data.observations).max(dim=1).values.flatten()
                # computing the TD Loss
                loss = F.mse_loss(td_target, value)
                # Network optimization via backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Target
            if global_step % tnuf == 0:
                # Copy the agent model into target network while using tnur
                for target_net_param, q_net_param in zip(target_net.parameters(), q_net.parameters()):
                    target_net_param.data.copy_(
                        tnur * q_net_param.data + (1.0 - tnur) * target_net_param.data
                    )
        if done:
            env.reset()
    
    # Visual Testing Env
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    done = False
    for step in range(1000):
        q_values = q_net(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values).cpu().numpy() # best action
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        if done:
            env.reset()
    env.close()
