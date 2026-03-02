#!/usr/bin/env python3
"""
Phase 2 Restart: Pick up from Block D2 using saved B2/C2 data.
Bug fix: KeyError in cross-method analysis (int→str key mismatch).
"""

import os, sys, json, time, copy, math, traceback, warnings
import numpy as np
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from utils.common import set_seed, ensure_dir

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'neurips_breakthrough')
SEEDS = [42, 137, 256]
DEVICE = torch.device('cpu')

from run_neurips_breakthrough import (
    ScalableCNN, ResNetCL, BasicBlock,
    get_architecture_zoo,
    load_cifar10_split, load_cifar100_split,
    run_cl_experiment, evaluate,
    compute_hessian_trace, compute_fisher_trace, compute_gradient_norm,
    compute_spectral_norm_approx, compute_metrics,
    estimate_eps_star, measure_intrinsic_curvature,
)

EPOCHS_PER_TASK = 5
FTR_CFG = {'epsilon': 0.2, 'lambda_init': 1.0, 'lambda_lr': 0.005,
           'lambda_max': 50.0, 'beta': 0.9, 'temperature': 2.0, 'warmup_epochs': 1}
METHOD_ARCHS = ['CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_D4_W32']


def main():
    print(f"Phase 2 RESTART — Block D2 onwards — Started {datetime.now()}")
    ensure_dir(RESULTS_DIR)
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    ensure_dir(plots_dir)

    zoo = get_architecture_zoo()

    # Load saved data
    with open(os.path.join(RESULTS_DIR, 'block_a_curvature.json')) as f:
        curvature_data = json.load(f)
    print(f"Loaded Block A: {len(curvature_data)} architectures")

    with open(os.path.join(RESULTS_DIR, 'block_b2_eps_star.json')) as f:
        eps_star_data = json.load(f)
    print(f"Loaded Block B2: {len(eps_star_data)} architectures, all ε*={list(set(d['eps_star'] for d in eps_star_data.values()))}")

    with open(os.path.join(RESULTS_DIR, 'block_c2_cifar100.json')) as f:
        c2_data = json.load(f)
        c100_curvature = c2_data['curvature']
        c100_eps_star = c2_data['eps_star']
    print(f"Loaded Block C2: {len(c100_eps_star)} architectures")
    for a, d in c100_eps_star.items():
        print(f"  {a}: ε*={d['eps_star']:.3f}")

    tasks_c10 = load_cifar10_split(5, 256, 1000)

    # ══════════════════════════════════════════════════════════════
    # BLOCK D2: CROSS-METHOD VALIDATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK D2: CROSS-METHOD VALIDATION")
    print("="*70)

    method_archs = [a for a in METHOD_ARCHS if a in zoo]
    method_grids = {
        'ewc': {'param': 'ewc_lambda', 'values': [1, 10, 100, 500, 1000, 5000, 10000]},
        'lwf': {'param': 'lwf_alpha', 'values': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        'si':  {'param': 'si_c', 'values': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
    }

    cross_method_data = {}
    for method_name, grid_cfg in method_grids.items():
        cross_method_data[method_name] = {}
        for arch_name in method_archs:
            arch_cfg = zoo[arch_name]
            arch_results = {}
            for hyper_val in grid_cfg['values']:
                hyper_results = []
                for seed in SEEDS[:2]:
                    cfg = {grid_cfg['param']: hyper_val, 'temperature': 2.0}
                    t0 = time.time()
                    print(f"  [{method_name}] {arch_name} {grid_cfg['param']}={hyper_val} seed={seed}",
                          end=" ", flush=True)
                    try:
                        r = run_cl_experiment(tasks_c10, arch_cfg['factory'], method_name, seed, DEVICE,
                                              epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
                        hyper_results.append(r)
                        print(f"✓ AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.0f}s)")
                    except Exception as e:
                        print(f"✗ {e}")
                if hyper_results:
                    arch_results[str(hyper_val)] = {
                        'forgetting': [r['forgetting'] for r in hyper_results],
                        'avg_accuracy': [r['average_accuracy'] for r in hyper_results],
                    }

            if arch_results:
                # BUG FIX: use original keys to avoid int→float→str mismatch
                keys = list(arch_results.keys())
                h_vals = [float(k) for k in keys]
                fg_means = [float(np.mean(arch_results[k]['forgetting'])) for k in keys]
                # For these methods, higher hyper → less forgetting
                # Find transition using inverted scale
                inv_h = [1.0/(h+1e-10) for h in h_vals]
                h_star, sharpness = estimate_eps_star(inv_h, fg_means)
                h_star_actual = 1.0/h_star if h_star > 0 else h_vals[0]

                cross_method_data[method_name][arch_name] = {
                    'hyper_values': h_vals,
                    'forgetting_means': fg_means,
                    'h_star': h_star_actual,
                    'sharpness': sharpness,
                }
                print(f"  → {method_name}/{arch_name}: h* = {h_star_actual:.2f}, sharpness = {sharpness:.2f}")

        # Incremental save per method
        with open(os.path.join(RESULTS_DIR, 'block_d2_cross_method.json'), 'w') as f:
            json.dump(cross_method_data, f, indent=2)
        print(f"  [{method_name}] saved. ({datetime.now()})")

    print(f"\nBlock D2 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # BLOCK E2: SCALING LAW ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK E2: SCALING LAW ANALYSIS")
    print("="*70)

    from run_neurips_breakthrough import run_scaling_analysis
    scaling_results = run_scaling_analysis(curvature_data, eps_star_data,
                                           c100_curvature, c100_eps_star)

    with open(os.path.join(RESULTS_DIR, 'block_e2_scaling.json'), 'w') as f:
        json.dump(scaling_results, f, indent=2)
    print(f"\nBlock E2 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # BLOCK F2: PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK F2: GENERATING PLOTS")
    print("="*70)

    from run_neurips_breakthrough import generate_plots
    generate_plots(curvature_data, eps_star_data, c100_curvature, c100_eps_star,
                   cross_method_data, scaling_results, plots_dir)
    print(f"\nBlock F2 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # BLOCK G2: DOSSIER
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK G2: GENERATING DOSSIER")
    print("="*70)

    from run_neurips_breakthrough import generate_dossier
    generate_dossier(curvature_data, eps_star_data, c100_curvature, c100_eps_star,
                     cross_method_data, scaling_results)
    print(f"\nBlock G2 done. ({datetime.now()})")

    print(f"\n{'='*70}")
    print(f"ALL BLOCKS COMPLETE. Finished: {datetime.now()}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
