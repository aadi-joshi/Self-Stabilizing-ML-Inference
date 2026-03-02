#!/usr/bin/env python3
"""
Supplementary: Extended-ε experiments to capture the full phase transition.
Uses 5 epochs/task and ε up to 200 to ensure we capture catastrophic regime.
Run AFTER the main breakthrough suite finishes (or in parallel).
"""

import os, sys, json, time, copy, math, traceback, warnings
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from utils.common import set_seed, ensure_dir

# Import from main script
from run_neurips_breakthrough import (
    ScalableCNN, ResNetCL, get_architecture_zoo,
    load_cifar10_split, run_cl_experiment, evaluate,
    compute_hessian_trace, compute_fisher_trace, compute_gradient_norm,
    compute_spectral_norm_approx, estimate_eps_star, compute_metrics,
    DEVICE, SEEDS
)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'neurips_breakthrough')
ensure_dir(RESULTS_DIR)

# Extended ε grid that captures the FULL transition
EPS_EXTENDED = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Key architectures covering the param range
KEY_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_W48', 'CNN_W64', 'CNN_W96',
             'CNN_W128', 'CNN_D4_W32', 'CNN_W32_NoBN',
             'ResNet18_W8', 'ResNet18_W16']

FTR_CFG = {'epsilon': 0.2, 'lambda_init': 1.0, 'lambda_lr': 0.005,
           'lambda_max': 50.0, 'lambda_momentum': 0.9, 'temperature': 2.0,
           'warmup_epochs': 1}

def main():
    print(f"Supplementary Extended-ε Sweep — Started {datetime.now()}")
    zoo = get_architecture_zoo()
    tasks = load_cifar10_split(5, 256, 1000)

    arch_list = [a for a in KEY_ARCHS if a in zoo]
    print(f"Architectures: {len(arch_list)}")
    print(f"ε grid: {EPS_EXTENDED}")
    print(f"Epochs per task: 5")
    print(f"Seeds: {SEEDS}")

    results = {}
    total = len(arch_list) * len(EPS_EXTENDED) * len(SEEDS)
    count = 0

    for arch_name in arch_list:
        arch_cfg = zoo[arch_name]
        arch_sweep = {}

        for eps in EPS_EXTENDED:
            eps_results = []
            for seed in SEEDS:
                count += 1
                cfg = dict(FTR_CFG)
                cfg['epsilon'] = eps
                t0 = time.time()
                print(f"  [{count}/{total}] {arch_name} eps={eps} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', seed, DEVICE,
                                          epochs_per_task=5, method_cfg=cfg)
                    eps_results.append(r)
                    print(f"✓ AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"✗ {e}")
                    traceback.print_exc()

            if eps_results:
                arch_sweep[str(eps)] = {
                    'avg_accuracy': [r['average_accuracy'] for r in eps_results],
                    'forgetting': [r['forgetting'] for r in eps_results],
                }

        if arch_sweep:
            eps_vals = [float(k) for k in arch_sweep.keys()]
            fg_means = [float(np.mean(arch_sweep[str(e)]['forgetting'])) for e in eps_vals]
            fg_stds = [float(np.std(arch_sweep[str(e)]['forgetting'], ddof=1))
                       if len(arch_sweep[str(e)]['forgetting']) > 1 else 0
                       for e in eps_vals]
            aa_means = [float(np.mean(arch_sweep[str(e)]['avg_accuracy'])) for e in eps_vals]
            aa_stds = [float(np.std(arch_sweep[str(e)]['avg_accuracy'], ddof=1))
                       if len(arch_sweep[str(e)]['avg_accuracy']) > 1 else 0
                       for e in eps_vals]

            e_star, sharpness = estimate_eps_star(eps_vals, fg_means)

            results[arch_name] = {
                'epsilon_values': eps_vals,
                'forgetting_means': fg_means,
                'forgetting_stds': fg_stds,
                'accuracy_means': aa_means,
                'accuracy_stds': aa_stds,
                'eps_star': e_star,
                'transition_sharpness': sharpness,
                'n_params': arch_cfg['n_params'],
            }
            print(f"  → {arch_name}: ε* = {e_star:.3f}, sharpness = {sharpness:.2f}")

        # Save incrementally
        with open(os.path.join(RESULTS_DIR, 'supplementary_extended_eps.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {datetime.now()}")
    print(f"Results: {os.path.join(RESULTS_DIR, 'supplementary_extended_eps.json')}")

    # Print summary
    print("\nSummary:")
    print(f"{'Architecture':<18} {'Params':>9} {'ε*':>8} {'Sharpness':>10} {'F(low)':>8} {'F(high)':>8}")
    for arch, d in sorted(results.items(), key=lambda x: x[1]['n_params']):
        fg = d['forgetting_means']
        ev = d['epsilon_values']
        below = [fg[i] for i in range(len(ev)) if ev[i] <= d['eps_star']]
        above = [fg[i] for i in range(len(ev)) if ev[i] > d['eps_star']]
        print(f"  {arch:<16} {d['n_params']:>9,} {d['eps_star']:>8.3f} {d['transition_sharpness']:>10.2f} "
              f"{np.mean(below):>8.3f} {np.mean(above) if above else 0:>8.3f}")


if __name__ == '__main__':
    main()
