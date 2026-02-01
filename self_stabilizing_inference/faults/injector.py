import random
import time
import torch

class FaultInjector:
    def __init__(self, latency_prob, noise_prob):
        self.latency_prob = latency_prob
        self.noise_prob = noise_prob

    def inject(self, x):
        if random.random() < self.latency_prob:
            time.sleep(0.05)

        if random.random() < self.noise_prob:
            x = x + 0.1 * torch.randn_like(x)

        return x
