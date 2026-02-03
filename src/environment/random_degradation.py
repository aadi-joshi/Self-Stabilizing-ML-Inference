import numpy as np
from .degradation_interface import DegradationProcess

class RandomDegradation(DegradationProcess):
    """
    Unseen random degradation generator for validation.
    Produces unpredictable noise and latency patterns not used during controller tuning.
    """
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.base_noise = 0.01
        self.max_noise = 0.25
        self.base_latency = 1000
        self.max_latency = 12000

    def get_noise(self, step):
        # Random walk with jumps and resets
        if step == 0 or not hasattr(self, '_noise'):  # initialize
            self._noise = self.base_noise
        jump = self.rng.uniform(-0.05, 0.05)
        if self.rng.rand() < 0.1:
            jump += self.rng.uniform(-0.15, 0.15)
        self._noise = np.clip(self._noise + jump, self.base_noise, self.max_noise)
        # Occasional resets
        if self.rng.rand() < 0.02:
            self._noise = self.base_noise
        return float(self._noise)

    def get_latency_load(self, step):
        # Random spikes and drops
        if step == 0 or not hasattr(self, '_latency'):
            self._latency = self.base_latency
        spike = self.rng.randint(-500, 500)
        if self.rng.rand() < 0.05:
            spike += self.rng.randint(2000, 8000)
        self._latency = int(np.clip(self._latency + spike, self.base_latency, self.max_latency))
        # Occasional resets
        if self.rng.rand() < 0.01:
            self._latency = self.base_latency
        return self._latency
