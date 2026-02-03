from enum import Enum

class StabilityState(Enum):
    STABLE = 0
    DEGRADED = 1
    RECOVERING = 2

class DualSignalController:
    def __init__(self, reliability_threshold, latency_threshold, min_dwell_steps):
        self.reliability_threshold = reliability_threshold
        self.latency_threshold = latency_threshold
        self.min_dwell_steps = min_dwell_steps

    def decide(self, reliability, latency, state, step, last_switch_step, active_model):
        action = None
        new_state = state
        # Switch to robust if reliability drops OR latency exceeds threshold
        if active_model == 'fast':
            if (reliability < self.reliability_threshold or latency > self.latency_threshold) and (step - last_switch_step > self.min_dwell_steps):
                action = 'robust'
                new_state = StabilityState.DEGRADED
        # Switch back to fast only if BOTH reliability and latency recover
        elif active_model == 'robust':
            if (reliability > self.reliability_threshold and latency < self.latency_threshold) and (step - last_switch_step > self.min_dwell_steps):
                action = 'fast'
                new_state = StabilityState.RECOVERING
        # Hysteresis: remain in RECOVERING until next switch
        if new_state == StabilityState.RECOVERING and action == 'fast':
            new_state = StabilityState.STABLE
        return action, new_state
