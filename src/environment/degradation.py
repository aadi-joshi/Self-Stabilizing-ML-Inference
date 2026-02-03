
import numpy as np
from .degradation_interface import DegradationProcess

class EnvironmentDegradation(DegradationProcess):
    def __init__(self, config):
        self.config = config
        self.burst_period = config.get('burst_period', 100)
        self.burst_length = config.get('burst_length', 10)
        self.burst_noise = config.get('burst_noise', 0.3)
        self.drift_start = config.get('drift_start', 200)
        self.drift_rate = config.get('drift_rate', 0.0005)
        self.osc_start = config.get('osc_start', 350)
        self.osc_period = config.get('osc_period', 20)
        self.osc_amplitude = config.get('osc_amplitude', 0.12)
        self.base_noise = config.get('base_noise', 0.01)
        self.degraded_noise = config.get('degraded_noise', 0.15)
        self.recovered_noise = config.get('recovered_noise', 0.03)
        self.base_latency = config.get('base_latency', 1000)
        self.degraded_latency = config.get('degraded_latency', 10000)

    def get_noise(self, step):
        # Bursty failures
        if (step // self.burst_period) != ((step-1) // self.burst_period):
            self._burst_active = True
            self._burst_end = step + self.burst_length
        if hasattr(self, '_burst_active') and self._burst_active and step < self._burst_end:
            return self.burst_noise
        if hasattr(self, '_burst_active') and self._burst_active and step >= self._burst_end:
            self._burst_active = False

        # Gradual drift
        drift = 0.0
        if step >= self.drift_start:
            drift = (step - self.drift_start) * self.drift_rate

        # Adversarial oscillation
        osc = 0.0
        if step >= self.osc_start:
            osc = self.osc_amplitude * np.sin(2 * np.pi * (step - self.osc_start) / self.osc_period)

        # Phase-based noise
        if step < 150:
            base = self.base_noise
        elif step < 300:
            base = self.degraded_noise
        else:
            base = self.recovered_noise
        return float(np.clip(base + drift + osc, 0, 1))

    def get_latency_load(self, step):
        # Bursty failures
        if hasattr(self, '_burst_active') and self._burst_active and step < self._burst_end:
            return self.degraded_latency
        # Adversarial oscillation: spike latency in sync with noise oscillation
        if step >= self.osc_start:
            osc = self.osc_amplitude * np.sin(2 * np.pi * (step - self.osc_start) / self.osc_period)
            if osc > 0.08:
                return self.degraded_latency
        # Phase-based latency
        if 150 <= step < 300:
            return self.degraded_latency
        return self.base_latency
