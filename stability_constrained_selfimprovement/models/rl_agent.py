# ============================================================================
# RL Policy Network and Gridworld Environment (Experiment C)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class GridWorld:
    """
    Configurable gridworld environment with obstacles and a goal.
    
    State: (row, col) flattened to one-hot or integer
    Actions: 0=up, 1=down, 2=left, 3=right
    Reward: +1 at goal, -0.01 per step, -0.1 for hitting wall/obstacle
    """

    def __init__(self, size: int = 8, seed: Optional[int] = None):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.rng = np.random.RandomState(seed)

        # Fixed obstacle layout for reproducibility
        self.obstacles = set()
        n_obstacles = size  # Number of obstacles
        obs_rng = np.random.RandomState(seed if seed else 0)
        while len(self.obstacles) < n_obstacles:
            r, c = obs_rng.randint(0, size), obs_rng.randint(0, size)
            if (r, c) != (0, 0) and (r, c) != (size - 1, size - 1):
                self.obstacles.add((r, c))

        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_pos = (0, 0)
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return one-hot state vector."""
        state = np.zeros(self.n_states, dtype=np.float32)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (next_state, reward, done, info)."""
        self.steps += 1
        r, c = self.agent_pos
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
        nr, nc = r + dr, c + dc

        reward = -0.01  # Step penalty

        # Check boundaries
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            reward = -0.1
            nr, nc = r, c  # Stay in place
        elif (nr, nc) in self.obstacles:
            reward = -0.1
            nr, nc = r, c  # Stay in place

        self.agent_pos = (nr, nc)

        done = self.agent_pos == self.goal
        if done:
            reward = 1.0

        # Timeout
        if self.steps >= self.size * self.size * 2:
            done = True

        return self._get_state(), reward, done, {}


class PolicyNetwork(nn.Module):
    """
    MLP policy network for gridworld.
    
    Input: one-hot state vector (grid_size²)
    Output: action probabilities (4 actions)
    """

    def __init__(self, n_states: int, n_actions: int = 4, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        layers = []
        in_dim = n_states
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representations for drift measurement."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, n_states) state vector
        Returns:
            action_probs: (batch_size, n_actions)
            state_value: (batch_size, 1)
        """
        features = self.backbone(x)
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(features)
        return action_probs, state_value

    def get_action(self, state: np.ndarray, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, value = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


class RolloutBuffer:
    """Simple buffer for storing REINFORCE rollouts."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
