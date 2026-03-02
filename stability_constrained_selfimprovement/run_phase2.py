#!/usr/bin/env python3
"""
NeurIPS Phase 2: Extended ε sweep + analysis
=============================================
Uses saved Block A curvature data.
Runs 5 epochs/task with ε range [0.01, 200] to capture full transition.
Then does cross-dataset, cross-method, scaling analysis, plots, dossier.
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

# Import architecture and training components
from run_neurips_breakthrough import (
    ScalableCNN, ResNetCL, BasicBlock,
    get_architecture_zoo,
    load_cifar10_split, load_cifar100_split,
    run_cl_experiment, evaluate,
    compute_hessian_trace, compute_fisher_trace, compute_gradient_norm,
    compute_spectral_norm_approx, compute_metrics,
    estimate_eps_star, measure_intrinsic_curvature,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

# Extended ε range covering full transition
EPS_FULL = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
EPS_REDUCED = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 200.0]  # For large models

EPOCHS_PER_TASK = 5  # More epochs for clearer forgetting

FTR_CFG = {'epsilon': 0.2, 'lambda_init': 1.0, 'lambda_lr': 0.005,
           'lambda_max': 50.0, 'lambda_momentum': 0.9, 'temperature': 2.0,
           'warmup_epochs': 1}

# Primary archs for extended sweep (11 = covers width, depth, bn, resnet)
PRIMARY_ARCHS = [
    'CNN_W8', 'CNN_W16', 'CNN_W24', 'CNN_W32', 'CNN_W48', 'CNN_W64',
    'CNN_W96', 'CNN_D4_W32', 'CNN_W32_NoBN',
    'ResNet18_W8', 'ResNet18_W16'
]

# Reduced set for cross-dataset (CIFAR-100 is slower with 10 tasks)
CROSS_DATASET_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_D4_W32', 'CNN_W32_NoBN']

# Archs for cross-method validation
METHOD_ARCHS = ['CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_D4_W32']


def main():
    print(f"NeurIPS Phase 2 — Extended ε Sweep — Started {datetime.now()}")
    print(f"Device: {DEVICE}, Epochs/task: {EPOCHS_PER_TASK}")
    ensure_dir(RESULTS_DIR)
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    ensure_dir(plots_dir)

    zoo = get_architecture_zoo()

    # Load Block A curvature data
    curv_path = os.path.join(RESULTS_DIR, 'block_a_curvature.json')
    if os.path.exists(curv_path):
        with open(curv_path) as f:
            curvature_data = json.load(f)
        print(f"Loaded curvature data: {len(curvature_data)} architectures from Block A")
    else:
        print("WARNING: No Block A curvature data found. Running curvature measurement...")
        curvature_data = run_curvature_block(zoo)

    tasks_c10 = load_cifar10_split(5, 256, 1000)

    # ══════════════════════════════════════════════════════════════
    # BLOCK B2: EXTENDED ε SWEEP (CIFAR-10, 5 epochs)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK B2: EXTENDED ε SWEEP (CIFAR-10, 5 epochs/task)")
    print("="*70)

    eps_star_data = {}
    arch_list = [a for a in PRIMARY_ARCHS if a in zoo]

    for arch_name in arch_list:
        arch_cfg = zoo[arch_name]
        is_large = arch_cfg['n_params'] > 500000
        grid = EPS_REDUCED if is_large else EPS_FULL

        arch_sweep = {}
        for eps in grid:
            eps_results = []
            for seed in SEEDS:
                cfg = dict(FTR_CFG)
                cfg['epsilon'] = eps
                t0 = time.time()
                print(f"  [{arch_name}] eps={eps} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks_c10, arch_cfg['factory'], 'ftr', seed, DEVICE,
                                          epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
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
                       if len(arch_sweep[str(e)]['forgetting']) > 1 else 0 for e in eps_vals]
            aa_means = [float(np.mean(arch_sweep[str(e)]['avg_accuracy'])) for e in eps_vals]
            aa_stds = [float(np.std(arch_sweep[str(e)]['avg_accuracy'], ddof=1))
                       if len(arch_sweep[str(e)]['avg_accuracy']) > 1 else 0 for e in eps_vals]

            e_star, sharpness = estimate_eps_star(eps_vals, fg_means)

            eps_star_data[arch_name] = {
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
            print(f"    F range: {min(fg_means):.3f} → {max(fg_means):.3f}")

        # Incremental save
        with open(os.path.join(RESULTS_DIR, 'block_b2_eps_star.json'), 'w') as f:
            json.dump(eps_star_data, f, indent=2)

    print(f"\nBlock B2 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # BLOCK C2: CROSS-DATASET (CIFAR-100)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BLOCK C2: CROSS-DATASET VALIDATION (CIFAR-100)")
    print("="*70)

    tasks_c100 = load_cifar100_split(10, 256, 400)
    c100_archs = [a for a in CROSS_DATASET_ARCHS if a in zoo]

    # Curvature on CIFAR-100
    c100_curvature = {}
    print("  --- Curvature measurement (CIFAR-100) ---")
    for arch_name in c100_archs:
        arch_cfg = zoo[arch_name]
        curvs = []
        for seed in SEEDS[:2]:  # 2 seeds to save time
            t0 = time.time()
            print(f"  [curv] {arch_name} seed={seed}", end=" ", flush=True)
            try:
                c = measure_intrinsic_curvature(
                    arch_cfg['factory'], tasks_c100, seed, DEVICE,
                    epochs=3, n_hutch=10, n_fisher_batches=5)
                curvs.append(c)
                print(f"✓ ht={c['hessian_trace']:.1f} ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"✗ {e}")
        if curvs:
            c100_curvature[arch_name] = {
                'n_params': curvs[0]['n_params'],
                'hessian_trace': float(np.mean([c['hessian_trace'] for c in curvs])),
                'fisher_trace': float(np.mean([c['fisher_trace'] for c in curvs])),
                'spectral_norm': float(np.mean([c['spectral_norm'] for c in curvs])),
                'd_eff': float(np.mean([c['d_eff'] for c in curvs])),
            }

    # ε sweep on CIFAR-100
    c100_eps_star = {}
    c100_grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 200.0]
    print("  --- ε sweep (CIFAR-100) ---")
    for arch_name in c100_archs:
        arch_cfg = zoo[arch_name]
        arch_sweep = {}
        for eps in c100_grid:
            eps_results = []
            for seed in SEEDS[:2]:
                cfg = dict(FTR_CFG); cfg['epsilon'] = eps
                t0 = time.time()
                print(f"  [{arch_name}] C100 eps={eps} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks_c100, arch_cfg['factory'], 'ftr', seed, DEVICE,
                                          epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
                    eps_results.append(r)
                    print(f"✓ AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"✗ {e}")
            if eps_results:
                arch_sweep[str(eps)] = {
                    'forgetting': [r['forgetting'] for r in eps_results],
                    'avg_accuracy': [r['average_accuracy'] for r in eps_results],
                }
        if arch_sweep:
            eps_vals = [float(k) for k in arch_sweep.keys()]
            fg_means = [float(np.mean(arch_sweep[str(e)]['forgetting'])) for e in eps_vals]
            e_star, sharpness = estimate_eps_star(eps_vals, fg_means)
            c100_eps_star[arch_name] = {
                'eps_star': e_star,
                'sharpness': sharpness,
                'forgetting_means': fg_means,
                'epsilon_values': eps_vals,
            }
            print(f"  → {arch_name} (C100): ε* = {e_star:.3f}")

    with open(os.path.join(RESULTS_DIR, 'block_c2_cifar100.json'), 'w') as f:
        json.dump({'curvature': c100_curvature, 'eps_star': c100_eps_star}, f, indent=2)
    print(f"\nBlock C2 done. ({datetime.now()})")

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

    with open(os.path.join(RESULTS_DIR, 'block_d2_cross_method.json'), 'w') as f:
        json.dump(cross_method_data, f, indent=2)
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


def run_curvature_block(zoo):
    """Fallback: re-run Block A if data not found."""
    curvature_data = {}
    tasks_c10 = load_cifar10_split(5, 256, 1000)
    for arch_name, arch_cfg in zoo.items():
        arch_curvatures = []
        for seed in SEEDS:
            try:
                c = measure_intrinsic_curvature(
                    arch_cfg['factory'], tasks_c10, seed, DEVICE,
                    epochs=arch_cfg['epochs'], n_hutch=10, n_fisher_batches=10)
                arch_curvatures.append(c)
            except Exception as e:
                print(f"  ✗ {arch_name} seed={seed}: {e}")
        if arch_curvatures:
            curvature_data[arch_name] = {
                'n_params': arch_curvatures[0]['n_params'],
                'hessian_trace': {'mean': float(np.mean([c['hessian_trace'] for c in arch_curvatures])),
                                  'std': float(np.std([c['hessian_trace'] for c in arch_curvatures], ddof=1)) if len(arch_curvatures) > 1 else 0},
                'fisher_trace': {'mean': float(np.mean([c['fisher_trace'] for c in arch_curvatures])),
                                 'std': float(np.std([c['fisher_trace'] for c in arch_curvatures], ddof=1)) if len(arch_curvatures) > 1 else 0},
                'spectral_norm': {'mean': float(np.mean([c['spectral_norm'] for c in arch_curvatures])),
                                  'std': float(np.std([c['spectral_norm'] for c in arch_curvatures], ddof=1)) if len(arch_curvatures) > 1 else 0},
                'd_eff': {'mean': float(np.mean([c['d_eff'] for c in arch_curvatures])),
                          'std': float(np.std([c['d_eff'] for c in arch_curvatures], ddof=1)) if len(arch_curvatures) > 1 else 0},
                'gradient_norm': {'mean': float(np.mean([c['gradient_norm'] for c in arch_curvatures]))},
                'task0_accuracy': float(np.mean([c['task0_accuracy'] for c in arch_curvatures])),
                'group': zoo[arch_name]['group'],
            }
    with open(os.path.join(RESULTS_DIR, 'block_a_curvature.json'), 'w') as f:
        json.dump(curvature_data, f, indent=2)
    return curvature_data


if __name__ == '__main__':
    main()
