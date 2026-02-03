from enum import Enum

class StabilityState(Enum):
    STABLE = 0
    DEGRADED = 1
    RECOVERING = 2
    PREEMPTIVE_DEGRADED = 3


class DualSignalController:
    def __init__(self, alpha, beta, gamma, horizon=1, min_dwell_steps=0, osc_window=10, osc_threshold=3, dwell_increase=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.horizon = horizon
        self.base_dwell = min_dwell_steps
        self.min_dwell_steps = min_dwell_steps
        self.switch_history = []
        self.last_switch_step = -100
        self.oscillation_count = 0
        self.osc_window = osc_window
        self.osc_threshold = osc_threshold
        self.dwell_increase = dwell_increase
        self.oscillating = False
        self.osc_start_step = None
        self.stabilization_time = None

    def _switch_penalty(self, step, active_model):
        # Penalty increases with recent oscillation
        penalty = self.gamma
        if len(self.switch_history) >= 4:
            # Count switches in last 4 steps
            recent = self.switch_history[-4:]
            if recent.count('switch') >= 2:
                penalty *= 2  # Double penalty if oscillating
        return penalty

    def _detect_oscillation(self, step):
        # Sliding window over switch_history
        window = self.switch_history[-self.osc_window:]
        switch_count = window.count('switch')
        if switch_count >= self.osc_threshold:
            if not self.oscillating:
                self.oscillating = True
                self.osc_start_step = step
            self.min_dwell_steps = self.base_dwell + self.dwell_increase
        else:
            if self.oscillating:
                self.oscillating = False
                if self.osc_start_step is not None:
                    self.stabilization_time = step - self.osc_start_step
                self.osc_start_step = None
            self.min_dwell_steps = self.base_dwell

    def decide(self, reliability, latency, state, step, last_switch_step, active_model,
               fast_pred=None, robust_pred=None):
        # Oscillation detection and adaptive dwell
        self._detect_oscillation(step)
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
        if len(self.switch_history) > 50:
            self.switch_history = self.switch_history[-50:]
        return action, new_state, self.oscillating, self.stabilization_time
