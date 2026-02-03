import numpy as np
from enum import Enum

class LearningControllerState(Enum):
    STABLE = 0
    DEGRADED = 1
    RECOVERING = 2
    PREEMPTIVE_DEGRADED = 3

class LearningController:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.99, seed=None):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}       # State-action value table
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.rng = np.random.RandomState(seed)

    def get_state(self, smoothed_reliability, smoothed_latency, rel_deriv, lat_deriv, osc_score, ctrl_state):
        # Discretize state for tabular Q-learning
        state = (
            round(smoothed_reliability, 2),
            round(smoothed_latency, 2),
            round(rel_deriv, 2),
            round(lat_deriv, 2),
            round(osc_score, 2),
            int(ctrl_state.value) if isinstance(ctrl_state, Enum) else int(ctrl_state)
        )
        return state

    def decide(self, smoothed_reliability, smoothed_latency, rel_deriv, lat_deriv, osc_score, ctrl_state):
        state = self.get_state(smoothed_reliability, smoothed_latency, rel_deriv, lat_deriv, osc_score, ctrl_state)
        self.last_state = state
        # Epsilon-greedy action selection
        if self.rng.rand() < self.epsilon or state not in self.q_table:
            action = self.rng.choice(['fast', 'robust'])
        else:
            q_vals = self.q_table[state]
            action = max(q_vals, key=q_vals.get)
        self.last_action = action
        self.state_history.append(state)
        self.action_history.append(action)
        return action, ctrl_state, False, None  # Oscillation, stabilization_time placeholders

    def update(self, reward, next_state):
        # Q-learning update
        state = self.last_state
        action = self.last_action
        if state not in self.q_table:
            self.q_table[state] = {'fast': 0.0, 'robust': 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {'fast': 0.0, 'robust': 0.0}
        best_next = max(self.q_table[next_state].values())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.reward_history.append(reward)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
