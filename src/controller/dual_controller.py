from enum import Enum

class StabilityState(Enum):
    STABLE = 0
    DEGRADED = 1
    RECOVERING = 2
    PREEMPTIVE_DEGRADED = 3

class DualSignalController:
    def __init__(self, alpha, beta, gamma, horizon=1, min_dwell_steps=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.horizon = horizon
        self.min_dwell_steps = min_dwell_steps
        self.switch_history = []
        self.last_switch_step = -100
        self.oscillation_count = 0

    def _switch_penalty(self, step, active_model):
        # Penalty increases with recent oscillation
        penalty = self.gamma
        if len(self.switch_history) >= 4:
            # Count switches in last 4 steps
            recent = self.switch_history[-4:]
            if recent.count('switch') >= 2:
                penalty *= 2  # Double penalty if oscillating
        return penalty

    def decide(self, reliability, latency, state, step, last_switch_step, active_model,
               fast_pred=None, robust_pred=None):
        # fast_pred, robust_pred: tuples (reliability, latency) for horizon prediction (optional)
        # If not provided, use current values
        if fast_pred is None:
            fast_pred = (reliability, latency)
        if robust_pred is None:
            robust_pred = (reliability, latency)

        # Compute cost for both models
        fast_J = self.alpha * (1 - fast_pred[0]) + self.beta * fast_pred[1]
        robust_J = self.alpha * (1 - robust_pred[0]) + self.beta * robust_pred[1]

        # Add switch penalty if switching
        penalty = self._switch_penalty(step, active_model)
        if active_model == 'fast' and robust_J + penalty < fast_J:
            action = 'robust'
            new_state = StabilityState.DEGRADED
            self.switch_history.append('switch')
            self.last_switch_step = step
        elif active_model == 'robust' and fast_J + penalty < robust_J:
            action = 'fast'
            new_state = StabilityState.RECOVERING
            self.switch_history.append('switch')
            self.last_switch_step = step
        else:
            action = None
            new_state = state
            self.switch_history.append('hold')

        # Hysteresis: remain in RECOVERING until next switch
        if new_state == StabilityState.RECOVERING and action == 'fast':
            new_state = StabilityState.STABLE
        # Keep switch_history bounded
        if len(self.switch_history) > 20:
            self.switch_history = self.switch_history[-20:]
        return action, new_state
