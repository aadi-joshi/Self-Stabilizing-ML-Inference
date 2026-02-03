import time
import torch

class LatencyMetric:
    def measure(self, model, x):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(torch.tensor(x, dtype=torch.float32).reshape(1, -1))
        latency = time.perf_counter() - start
        return latency
