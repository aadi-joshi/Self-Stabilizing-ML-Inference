def run(system, adaptive=True):
    reliability = []
    for step in range(500):
        r, degraded, action = system.step(adaptive=adaptive)
        reliability.append(r)
    return reliability

from experiments.system import SelfStabilizingSystem
import numpy as np

adaptive_scores = run(system, adaptive=True)
baseline_scores = run(system, adaptive=False)

print("Adaptive mean reliability:", np.mean(adaptive_scores))
print("Baseline mean reliability:", np.mean(baseline_scores))

