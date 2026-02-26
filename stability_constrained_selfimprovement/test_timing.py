#!/usr/bin/env python3
"""Quick timing test: 1 experiment on CPU."""
import sys, time
sys.path.insert(0, '.')
import torch
from run_complete import run_experiment

device = torch.device('cpu')
print(f'Device: {device}')

t0 = time.time()
r = run_experiment('split_cifar10', 'baseline', 42, device, epochs_per_task=5)
elapsed = time.time() - t0
print(f'Done in {elapsed:.1f}s')
print(f'AA={r["average_accuracy"]:.3f} F={r["forgetting"]:.3f}')
