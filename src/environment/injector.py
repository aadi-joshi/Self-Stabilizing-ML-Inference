import random
import torch
import time

class FaultInjector:
    def __init__(self, latency_prob, noise_prob):
        self.latency_prob = latency_prob
        self.noise_prob = noise_prob
    def inject(self, x, step=None, env=None):
        import numpy as np
        import torch
        # Simulate latency degradation based on environment phase
        latency_load = env.get_latency_load(step) if (env and step is not None) else 1000
        if random.random() < self.latency_prob:
            _ = [sum([i**2 for i in range(latency_load)]) for _ in range(2)]
        # Ensure x is a torch tensor for noise injection
        if random.random() < self.noise_prob:
            if isinstance(x, np.ndarray):
                x_t = torch.tensor(x, dtype=torch.float32)
                x_t = x_t + 0.2 * torch.randn_like(x_t)
                x = x_t.numpy()
            elif isinstance(x, torch.Tensor):
                x = x + 0.2 * torch.randn_like(x)
        return x
