#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '.')
import torch
from run_fast import run_experiment

device = torch.device('cpu')
print(f'Device: {device}', flush=True)

t0 = time.time()
r = run_experiment('split_cifar10', 'baseline', 42, device, epochs_per_task=5)
elapsed = time.time() - t0
print(f'baseline: {elapsed:.1f}s AA={r["average_accuracy"]:.3f} F={r["forgetting"]:.3f}', flush=True)

t0 = time.time()
r = run_experiment('split_cifar10', 'ftr', 42, device, epochs_per_task=5,
                   method_cfg={'epsilon': 0.2, 'lambda_init': 1.0, 'lambda_lr': 0.005,
                               'lambda_max': 50.0, 'lambda_momentum': 0.9, 'temperature': 2.0, 'warmup_epochs': 1})
elapsed = time.time() - t0
print(f'ftr: {elapsed:.1f}s AA={r["average_accuracy"]:.3f} F={r["forgetting"]:.3f}', flush=True)
print('DONE', flush=True)
