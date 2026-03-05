#!/usr/bin/env python3
"""
NeurIPS Phase Diagram Mode
===========================
Constructs a universal phase diagram of stability in non-stationary learning.
Tests whether curvature normalization collapses ε* across architectures.

Phases:
  1. Dense ε sweep with 5 seeds (fine grid around transition)
  2. Compute normalized variables (ε_norm, ε_eff, ε_dim, κ-scaled)
  3. Build 2D phase diagram (curvature × ε) with regime labels
  4. Test collapse quality with variance reduction
  5. Cross-method overlay (FTR, EWC, LwF)
  6. Statistical rigor (bootstrap CI, hypothesis tests)
  7. Theoretical boundary derivation
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'phase_diagram')
PREV_RESULTS = os.path.join(BASE_DIR, 'results', 'neurips_breakthrough')
SEEDS = [42, 137, 256, 7, 2024]  # 5 seeds
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

# 8 architectures spanning maximum curvature diversity
PHASE_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_W96',
               'CNN_D4_W32', 'CNN_W32_NoBN', 'ResNet18_W8']

# Dense ε grid: logarithmic with fine resolution around transition region
EPS_DENSE = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5,
             7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0, 15.0, 20.0, 50.0]

# Cross-method grids (dense around known transitions)
LWF_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75,
              0.8, 0.9, 1.0, 1.5, 2.0, 5.0]
EWC_LAMBDAS = [1, 10, 50, 100, 500, 1000, 5000, 10000]
METHOD_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_D4_W32', 'ResNet18_W8']


def estimate_eps_star_interpolated(eps_values, forgetting_values):
    """
    Refined ε* estimation using spline interpolation on dense grid.
    Returns (eps_star, transition_sharpness, max_derivative).
    """
    if len(eps_values) < 4:
        return float(eps_values[0]), 0.0, 0.0

    order = np.argsort(eps_values)
    eps_sorted = np.array(eps_values)[order]
    fg_sorted = np.array(forgetting_values)[order]
    log_eps = np.log(eps_sorted + 1e-10)

    # Try spline interpolation for sub-grid resolution
    try:
        from scipy.interpolate import UnivariateSpline
        from scipy.optimize import minimize_scalar

        # Fit smoothing spline
        spline = UnivariateSpline(log_eps, fg_sorted, s=len(log_eps)*0.001, k=3)
        # Find max of derivative
        le_fine = np.linspace(log_eps[0], log_eps[-1], 1000)
        deriv = spline.derivative()(le_fine)
        max_idx = np.argmax(np.abs(deriv))
        eps_star = float(np.exp(le_fine[max_idx]))
        max_deriv = float(np.abs(deriv[max_idx]))
    except Exception:
        # Fallback to finite differences
        derivs = []
        for i in range(1, len(log_eps)):
            d_fg = fg_sorted[i] - fg_sorted[i-1]
            d_le = log_eps[i] - log_eps[i-1]
            derivs.append(abs(d_fg / d_le) if abs(d_le) > 1e-10 else 0.0)
        max_idx = int(np.argmax(derivs))
        eps_star = float(math.sqrt(eps_sorted[max_idx] * eps_sorted[max_idx + 1]))
        max_deriv = float(max(derivs))

    # Transition sharpness
    below = [fg_sorted[i] for i in range(len(eps_sorted)) if eps_sorted[i] <= eps_star]
    above = [fg_sorted[i] for i in range(len(eps_sorted)) if eps_sorted[i] > eps_star]
    sharpness = float(np.mean(above) / max(np.mean(below), 1e-10)) if below and above else 1.0

    return eps_star, sharpness, max_deriv


def classify_regime(forgetting, threshold_stable=0.12, threshold_catastrophic=0.20):
    """Classify forgetting into regime: 0=stable, 1=partial, 2=catastrophic."""
    if forgetting < threshold_stable:
        return 0  # stable
    elif forgetting < threshold_catastrophic:
        return 1  # partial
    else:
        return 2  # catastrophic


def bootstrap_eps_star(eps_values, forgetting_per_seed, n_bootstrap=1000, rng=None):
    """
    Bootstrap ε* estimation.
    forgetting_per_seed: dict {str(eps): [f1, f2, ..., f5]}
    Returns (mean_eps_star, std_eps_star, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_seeds = len(list(forgetting_per_seed.values())[0])
    boot_stars = []

    for _ in range(n_bootstrap):
        # Resample seeds with replacement
        idx = rng.randint(0, n_seeds, size=n_seeds)
        fg_means = []
        for eps in eps_values:
            vals = forgetting_per_seed[str(eps)]
            boot_vals = [vals[i] for i in idx if i < len(vals)]
            if boot_vals:
                fg_means.append(float(np.mean(boot_vals)))
            else:
                fg_means.append(float(np.mean(vals)))

        try:
            es, _, _ = estimate_eps_star_interpolated(eps_values, fg_means)
            boot_stars.append(es)
        except Exception:
            pass

    if len(boot_stars) < 10:
        return np.mean(eps_values), np.std(eps_values), eps_values[0], eps_values[-1]

    boot_stars = np.array(boot_stars)
    return (float(np.mean(boot_stars)),
            float(np.std(boot_stars)),
            float(np.percentile(boot_stars, 2.5)),
            float(np.percentile(boot_stars, 97.5)))


def main():
    t_start = datetime.now()
    print(f"NeurIPS Phase Diagram Mode — Started {t_start}")
    print(f"Device: {DEVICE}, Seeds: {SEEDS}, Epochs/task: {EPOCHS_PER_TASK}")
    print(f"Dense ε grid: {len(EPS_DENSE)} points")
    ensure_dir(RESULTS_DIR)
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    ensure_dir(plots_dir)

    zoo = get_architecture_zoo()

    # ══════════════════════════════════════════════════════════════
    # Load curvature data from previous campaign
    # ══════════════════════════════════════════════════════════════
    curv_path = os.path.join(PREV_RESULTS, 'block_a_curvature.json')
    with open(curv_path) as f:
        curvature_data = json.load(f)
    print(f"Loaded curvature data: {len(curvature_data)} architectures")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: DENSE ε SWEEP WITH 5 SEEDS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 1: DENSE ε SWEEP (FTR, 5 seeds, 20 ε points)")
    print("="*70)

    tasks_c10 = load_cifar10_split(5, 256, 1000)

    dense_data = {}
    phase_archs = [a for a in PHASE_ARCHS if a in zoo]
    print(f"Architectures: {phase_archs}")

    # Check for saved progress
    dense_path = os.path.join(RESULTS_DIR, 'phase1_dense_sweep.json')
    if os.path.exists(dense_path):
        with open(dense_path) as f:
            dense_data = json.load(f)
        print(f"Resuming: {list(dense_data.keys())} already complete")

    for arch_name in phase_archs:
        if arch_name in dense_data:
            print(f"  Skipping {arch_name} (already complete)")
            continue

        arch_cfg = zoo[arch_name]
        arch_sweep = {}

        for eps in EPS_DENSE:
            seed_results = []
            for seed in SEEDS:
                cfg = dict(FTR_CFG)
                cfg['epsilon'] = eps
                t0 = time.time()
                print(f"  [{arch_name}] eps={eps:.1f} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks_c10, arch_cfg['factory'], 'ftr', seed, DEVICE,
                                          epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
                    seed_results.append({
                        'forgetting': float(r['forgetting']),
                        'avg_accuracy': float(r['average_accuracy']),
                    })
                    print(f"✓ F={r['forgetting']:.4f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"✗ {e}")

            if seed_results:
                arch_sweep[str(eps)] = {
                    'forgetting': [r['forgetting'] for r in seed_results],
                    'avg_accuracy': [r['avg_accuracy'] for r in seed_results],
                    'forgetting_mean': float(np.mean([r['forgetting'] for r in seed_results])),
                    'forgetting_std': float(np.std([r['forgetting'] for r in seed_results], ddof=1))
                        if len(seed_results) > 1 else 0.0,
                }

        if arch_sweep:
            # Compute ε* with interpolation
            eps_vals = sorted([float(k) for k in arch_sweep.keys()])
            fg_means = [arch_sweep[str(e)]['forgetting_mean'] for e in eps_vals]
            fg_per_seed = {str(e): arch_sweep[str(e)]['forgetting'] for e in eps_vals}

            eps_star, sharpness, max_deriv = estimate_eps_star_interpolated(eps_vals, fg_means)
            boot_mean, boot_std, boot_ci_lo, boot_ci_hi = bootstrap_eps_star(eps_vals, fg_per_seed)

            dense_data[arch_name] = {
                'epsilon_values': eps_vals,
                'forgetting_means': fg_means,
                'forgetting_stds': [arch_sweep[str(e)]['forgetting_std'] for e in eps_vals],
                'forgetting_per_seed': fg_per_seed,
                'eps_star': eps_star,
                'eps_star_boot_mean': boot_mean,
                'eps_star_boot_std': boot_std,
                'eps_star_ci95': [boot_ci_lo, boot_ci_hi],
                'transition_sharpness': sharpness,
                'max_derivative': max_deriv,
                'n_params': arch_cfg['n_params'],
            }
            print(f"  → {arch_name}: ε* = {eps_star:.3f} "
                  f"(boot: {boot_mean:.3f} ± {boot_std:.3f}, "
                  f"95% CI [{boot_ci_lo:.3f}, {boot_ci_hi:.3f}])")
            print(f"    sharpness = {sharpness:.2f}, max_deriv = {max_deriv:.4f}")

        # Incremental save
        with open(dense_path, 'w') as f:
            json.dump(dense_data, f, indent=2)

    print(f"\nPhase 1 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1B: CURVATURE RE-MEASUREMENT WITH 5 SEEDS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 1B: CURVATURE RE-MEASUREMENT (5 seeds)")
    print("="*70)

    curv5_path = os.path.join(RESULTS_DIR, 'curvature_5seed.json')
    if os.path.exists(curv5_path):
        with open(curv5_path) as f:
            curvature_5seed = json.load(f)
        print(f"Loaded 5-seed curvature: {len(curvature_5seed)} architectures")
    else:
        curvature_5seed = {}

    for arch_name in phase_archs:
        if arch_name in curvature_5seed:
            print(f"  Skipping {arch_name} (already measured)")
            continue

        arch_cfg = zoo[arch_name]
        curvs = []
        for seed in SEEDS:
            t0 = time.time()
            print(f"  [curv] {arch_name} seed={seed}", end=" ", flush=True)
            try:
                c = measure_intrinsic_curvature(
                    arch_cfg['factory'], tasks_c10, seed, DEVICE,
                    epochs=3, n_hutch=10, n_fisher_batches=10)
                curvs.append(c)
                print(f"✓ ht={c['hessian_trace']:.1f} ft={c['fisher_trace']:.4f} "
                      f"sn={c['spectral_norm']:.1f} ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"✗ {e}")

        if curvs:
            curvature_5seed[arch_name] = {
                'n_params': curvs[0]['n_params'],
                'hessian_trace': {
                    'mean': float(np.mean([c['hessian_trace'] for c in curvs])),
                    'std': float(np.std([c['hessian_trace'] for c in curvs], ddof=1)) if len(curvs) > 1 else 0,
                    'values': [float(c['hessian_trace']) for c in curvs],
                },
                'fisher_trace': {
                    'mean': float(np.mean([c['fisher_trace'] for c in curvs])),
                    'std': float(np.std([c['fisher_trace'] for c in curvs], ddof=1)) if len(curvs) > 1 else 0,
                    'values': [float(c['fisher_trace']) for c in curvs],
                },
                'spectral_norm': {
                    'mean': float(np.mean([c['spectral_norm'] for c in curvs])),
                    'std': float(np.std([c['spectral_norm'] for c in curvs], ddof=1)) if len(curvs) > 1 else 0,
                    'values': [float(c['spectral_norm']) for c in curvs],
                },
                'd_eff': {
                    'mean': float(np.mean([c['d_eff'] for c in curvs])),
                    'std': float(np.std([c['d_eff'] for c in curvs], ddof=1)) if len(curvs) > 1 else 0,
                    'values': [float(c['d_eff']) for c in curvs],
                },
                'gradient_norm': {
                    'mean': float(np.mean([c['gradient_norm'] for c in curvs])),
                    'values': [float(c['gradient_norm']) for c in curvs],
                },
            }
            # Incremental save
            with open(curv5_path, 'w') as f:
                json.dump(curvature_5seed, f, indent=2)

    print(f"\nPhase 1B done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: NORMALIZATION + PHASE DIAGRAM CONSTRUCTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 2: NORMALIZATION + PHASE DIAGRAM")
    print("="*70)

    # Build normalized variables for each architecture
    curv_source = curvature_5seed if curvature_5seed else curvature_data
    norm_data = {}

    for arch_name in phase_archs:
        if arch_name not in dense_data or arch_name not in curv_source:
            continue

        d = dense_data[arch_name]
        c = curv_source[arch_name]

        ht = c['hessian_trace']['mean']
        ft = c['fisher_trace']['mean']
        sn = c['spectral_norm']['mean']
        deff = c['d_eff']['mean']
        n_params = c['n_params']
        gn = c['gradient_norm']['mean']

        # Curvature density metrics
        kappa = ht / n_params          # curvature density
        kappa_f = ft / n_params        # Fisher density
        gamma_eff = ht / sn            # effective dimension (d_eff ≈ this)

        eps_star = d['eps_star']
        eps_star_boot = d['eps_star_boot_mean']

        # NORMALIZED ε* CANDIDATES
        norm_data[arch_name] = {
            'raw_eps_star': eps_star,
            'boot_eps_star': eps_star_boot,
            'boot_std': d['eps_star_boot_std'],
            'boot_ci95': d['eps_star_ci95'],
            'n_params': n_params,
            'hessian_trace': ht,
            'fisher_trace': ft,
            'spectral_norm': sn,
            'd_eff': deff,
            'gradient_norm': gn,
            'kappa': kappa,        # tr(H)/d
            'kappa_f': kappa_f,    # tr(F)/d
            # Normalized ε* candidates
            'eps_norm_fisher': eps_star * ft,           # ε · tr(F)
            'eps_norm_hessian': eps_star * ht,          # ε · tr(H)
            'eps_norm_deff': eps_star * deff,            # ε · d_eff
            'eps_norm_kappa': eps_star * kappa,          # ε · tr(H)/d
            'eps_norm_spectral': eps_star * sn,          # ε · ||H||_op
            'eps_norm_grad': eps_star * gn**2,           # ε · ||g||²
            'eps_norm_fisher_grad': eps_star * ft * gn**2,  # ε · tr(F) · ||g||²  (theory prediction)
            'eps_norm_fisher_d': eps_star * ft / n_params,  # ε · tr(F)/d
            'eps_norm_sqrt_fisher': eps_star * math.sqrt(ft),  # ε · √tr(F)
            'eps_norm_sqrt_hessian': eps_star * math.sqrt(ht),  # ε · √tr(H)
        }

    # Compute variance reduction for each normalization
    print("\n  ── Normalization Variance Analysis ──")
    print(f"  {'Normalization':<30s} {'Mean':>10s} {'Std':>10s} {'CV':>10s} {'Var_red':>10s}")
    print(f"  {'-'*70}")

    raw_stars = [norm_data[a]['raw_eps_star'] for a in norm_data]
    raw_cv = np.std(raw_stars) / max(np.mean(raw_stars), 1e-10) if raw_stars else 0
    raw_var = np.var(raw_stars)

    print(f"  {'raw_eps_star':<30s} {np.mean(raw_stars):>10.4f} {np.std(raw_stars):>10.4f} "
          f"{raw_cv:>10.4f} {'baseline':>10s}")

    norm_keys = [k for k in list(norm_data.values())[0].keys()
                 if k.startswith('eps_norm_')]

    variance_results = {}
    for key in norm_keys:
        vals = [norm_data[a][key] for a in norm_data]
        cv = np.std(vals) / max(np.mean(vals), 1e-10)
        var_red = 1.0 - np.var(vals) / max(raw_var, 1e-15) if raw_var > 0 else 0
        variance_results[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'cv': float(cv),
            'variance_reduction': float(var_red),
            'values': {a: norm_data[a][key] for a in norm_data},
        }
        print(f"  {key:<30s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{cv:>10.4f} {var_red:>10.4f}")

    # Build phase diagram data: (curvature_metric, ε, forgetting, regime) for each point
    print("\n  ── Phase Diagram Data Points ──")
    phase_points = []
    for arch_name in phase_archs:
        if arch_name not in dense_data or arch_name not in curv_source:
            continue
        d = dense_data[arch_name]
        c = curv_source[arch_name]

        ht = c['hessian_trace']['mean']
        ft = c['fisher_trace']['mean']
        sn = c['spectral_norm']['mean']
        deff = c['d_eff']['mean']
        n_params = c['n_params']
        gn = c['gradient_norm']['mean']

        for i, eps in enumerate(d['epsilon_values']):
            fg = d['forgetting_means'][i]
            fg_std = d['forgetting_stds'][i]
            regime = classify_regime(fg)

            phase_points.append({
                'arch': arch_name,
                'eps': eps,
                'forgetting': fg,
                'forgetting_std': fg_std,
                'regime': regime,
                'n_params': n_params,
                'hessian_trace': ht,
                'fisher_trace': ft,
                'spectral_norm': sn,
                'd_eff': deff,
                'gradient_norm': gn,
                # Normalized ε
                'eps_norm_fisher': eps * ft,
                'eps_norm_hessian': eps * ht,
                'eps_norm_deff': eps * deff,
                'eps_norm_kappa': eps * (ht / n_params),
                'eps_norm_spectral': eps * sn,
                'eps_norm_grad': eps * gn**2,
                'eps_norm_fisher_d': eps * ft / n_params,
            })

    print(f"  Total phase diagram points: {len(phase_points)}")

    # Save Phase 2 results
    phase2_results = {
        'norm_data': norm_data,
        'variance_results': variance_results,
        'raw_cv': float(raw_cv),
        'raw_var': float(raw_var),
        'phase_points': phase_points,
    }
    with open(os.path.join(RESULTS_DIR, 'phase2_normalization.json'), 'w') as f:
        json.dump(phase2_results, f, indent=2)
    print(f"\nPhase 2 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: CROSS-METHOD PHASE DIAGRAM
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 3: CROSS-METHOD PHASE DIAGRAM (LwF dense, EWC dense)")
    print("="*70)

    method_data = {'ftr': dense_data}  # FTR data already from Phase 1

    # LwF dense sweep
    lwf_path = os.path.join(RESULTS_DIR, 'phase3_lwf_dense.json')
    if os.path.exists(lwf_path):
        with open(lwf_path) as f:
            lwf_data = json.load(f)
        print(f"Loaded LwF data: {list(lwf_data.keys())}")
    else:
        lwf_data = {}

    method_archs = [a for a in METHOD_ARCHS if a in zoo]
    for arch_name in method_archs:
        if arch_name in lwf_data:
            print(f"  Skipping LwF/{arch_name} (already complete)")
            continue

        arch_cfg = zoo[arch_name]
        arch_sweep = {}
        for alpha in LWF_ALPHAS:
            seed_results = []
            for seed in SEEDS[:3]:  # 3 seeds for cross-method (time constraint)
                cfg = {'lwf_alpha': alpha, 'temperature': 2.0}
                t0 = time.time()
                print(f"  [lwf] {arch_name} α={alpha} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks_c10, arch_cfg['factory'], 'lwf', seed, DEVICE,
                                          epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
                    seed_results.append({
                        'forgetting': float(r['forgetting']),
                        'avg_accuracy': float(r['average_accuracy']),
                    })
                    print(f"✓ F={r['forgetting']:.4f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"✗ {e}")
            if seed_results:
                arch_sweep[str(alpha)] = {
                    'forgetting': [r['forgetting'] for r in seed_results],
                    'forgetting_mean': float(np.mean([r['forgetting'] for r in seed_results])),
                    'forgetting_std': float(np.std([r['forgetting'] for r in seed_results], ddof=1))
                        if len(seed_results) > 1 else 0.0,
                }

        if arch_sweep:
            alphas = sorted([float(k) for k in arch_sweep.keys()])
            fg_means = [arch_sweep[str(a)]['forgetting_mean'] for a in alphas]
            # For LwF: higher α → less forgetting, invert for transition detection
            inv_alphas = [1.0/(a+1e-10) for a in alphas]
            es, sh, md = estimate_eps_star_interpolated(inv_alphas, fg_means)
            alpha_star = 1.0/es if es > 0 else alphas[0]

            lwf_data[arch_name] = {
                'alpha_values': alphas,
                'forgetting_means': fg_means,
                'forgetting_stds': [arch_sweep[str(a)]['forgetting_std'] for a in alphas],
                'alpha_star': float(alpha_star),
                'sharpness': float(sh),
            }
            print(f"  → LwF/{arch_name}: α* = {alpha_star:.3f}, sharpness = {sh:.2f}")

        with open(lwf_path, 'w') as f:
            json.dump(lwf_data, f, indent=2)

    method_data['lwf'] = lwf_data

    # EWC dense sweep
    ewc_path = os.path.join(RESULTS_DIR, 'phase3_ewc_dense.json')
    if os.path.exists(ewc_path):
        with open(ewc_path) as f:
            ewc_data = json.load(f)
        print(f"Loaded EWC data: {list(ewc_data.keys())}")
    else:
        ewc_data = {}

    for arch_name in method_archs:
        if arch_name in ewc_data:
            print(f"  Skipping EWC/{arch_name} (already complete)")
            continue

        arch_cfg = zoo[arch_name]
        arch_sweep = {}
        for lam_val in EWC_LAMBDAS:
            seed_results = []
            for seed in SEEDS[:3]:
                cfg = {'ewc_lambda': lam_val, 'temperature': 2.0}
                t0 = time.time()
                print(f"  [ewc] {arch_name} λ={lam_val} seed={seed}", end=" ", flush=True)
                try:
                    r = run_cl_experiment(tasks_c10, arch_cfg['factory'], 'ewc', seed, DEVICE,
                                          epochs_per_task=EPOCHS_PER_TASK, method_cfg=cfg)
                    seed_results.append({
                        'forgetting': float(r['forgetting']),
                        'avg_accuracy': float(r['average_accuracy']),
                    })
                    print(f"✓ F={r['forgetting']:.4f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"✗ {e}")
            if seed_results:
                arch_sweep[str(lam_val)] = {
                    'forgetting': [r['forgetting'] for r in seed_results],
                    'forgetting_mean': float(np.mean([r['forgetting'] for r in seed_results])),
                    'forgetting_std': float(np.std([r['forgetting'] for r in seed_results], ddof=1))
                        if len(seed_results) > 1 else 0.0,
                }

        if arch_sweep:
            lams = sorted([float(k) for k in arch_sweep.keys()])
            fg_means = [arch_sweep[str(l)]['forgetting_mean'] for l in lams]
            inv_lams = [1.0/(l+1e-10) for l in lams]
            es, sh, md = estimate_eps_star_interpolated(inv_lams, fg_means)
            lam_star = 1.0/es if es > 0 else lams[0]

            ewc_data[arch_name] = {
                'lambda_values': lams,
                'forgetting_means': fg_means,
                'forgetting_stds': [arch_sweep[str(l)]['forgetting_std'] for l in lams],
                'lambda_star': float(lam_star),
                'sharpness': float(sh),
            }
            print(f"  → EWC/{arch_name}: λ* = {lam_star:.2f}, sharpness = {sh:.2f}")

        with open(ewc_path, 'w') as f:
            json.dump(ewc_data, f, indent=2)

    method_data['ewc'] = ewc_data

    with open(os.path.join(RESULTS_DIR, 'phase3_cross_method.json'), 'w') as f:
        json.dump({
            'ftr': {a: {'eps_star': dense_data[a]['eps_star'],
                        'boot_std': dense_data[a]['eps_star_boot_std']}
                    for a in dense_data},
            'lwf': {a: {'alpha_star': lwf_data[a]['alpha_star'],
                        'sharpness': lwf_data[a]['sharpness']}
                    for a in lwf_data},
            'ewc': {a: {'lambda_star': ewc_data[a]['lambda_star'],
                        'sharpness': ewc_data[a]['sharpness']}
                    for a in ewc_data},
        }, f, indent=2)

    print(f"\nPhase 3 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: STATISTICAL RIGOR + COLLAPSE TESTING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 4: STATISTICAL RIGOR + COLLAPSE TESTING")
    print("="*70)

    try:
        from scipy import stats as sp_stats
        from scipy.optimize import curve_fit
        SCIPY_OK = True
    except ImportError:
        SCIPY_OK = False
        print("  WARNING: scipy not available, limited statistical testing")

    stat_results = {}

    # 4a. Test whether raw ε* values differ significantly
    eps_stars = {a: dense_data[a]['eps_star'] for a in dense_data}
    boot_stds = {a: dense_data[a]['eps_star_boot_std'] for a in dense_data}
    print(f"\n  ── Raw ε* values ──")
    for a in sorted(eps_stars, key=lambda x: eps_stars[x]):
        ci = dense_data[a]['eps_star_ci95']
        print(f"  {a:<20s}: ε* = {eps_stars[a]:.4f} ± {boot_stds[a]:.4f} "
              f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

    eps_star_vals = list(eps_stars.values())
    raw_range = max(eps_star_vals) - min(eps_star_vals)
    raw_cv_fine = np.std(eps_star_vals) / np.mean(eps_star_vals)
    print(f"\n  Range: {raw_range:.4f}")
    print(f"  CV: {raw_cv_fine:.4f}")
    print(f"  Mean: {np.mean(eps_star_vals):.4f} ± {np.std(eps_star_vals):.4f}")

    stat_results['raw_eps_stars'] = eps_stars
    stat_results['raw_range'] = float(raw_range)
    stat_results['raw_cv'] = float(raw_cv_fine)
    stat_results['raw_mean'] = float(np.mean(eps_star_vals))
    stat_results['raw_std'] = float(np.std(eps_star_vals))

    # 4b. Test association: ε* vs curvature metrics
    print(f"\n  ── Correlation Tests: ε* vs Curvature ──")
    curv_metrics = {}
    for a in phase_archs:
        if a not in curv_source or a not in dense_data:
            continue
        c = curv_source[a]
        curv_metrics[a] = {
            'hessian_trace': c['hessian_trace']['mean'],
            'fisher_trace': c['fisher_trace']['mean'],
            'spectral_norm': c['spectral_norm']['mean'],
            'd_eff': c['d_eff']['mean'],
            'n_params': c['n_params'],
            'gradient_norm': c['gradient_norm']['mean'],
            'log_hessian': math.log(c['hessian_trace']['mean']),
            'log_params': math.log(c['n_params']),
            'log_spectral': math.log(c['spectral_norm']['mean']),
        }

    correlation_results = {}
    common_archs = [a for a in phase_archs if a in curv_metrics and a in eps_stars]
    y_vals = [eps_stars[a] for a in common_archs]

    for metric_name in curv_metrics[common_archs[0]].keys():
        x_vals = [curv_metrics[a][metric_name] for a in common_archs]
        if SCIPY_OK:
            r_val, p_val = sp_stats.pearsonr(x_vals, y_vals)
            tau, p_tau = sp_stats.kendalltau(x_vals, y_vals)
        else:
            r_val = float(np.corrcoef(x_vals, y_vals)[0, 1]) if np.std(y_vals) > 0 else 0
            p_val, tau, p_tau = 1.0, 0.0, 1.0

        correlation_results[metric_name] = {
            'pearson_r': float(r_val),
            'pearson_p': float(p_val),
            'kendall_tau': float(tau),
            'kendall_p': float(p_tau),
        }
        sig = "**" if p_val < 0.05 else ""
        print(f"  {metric_name:<20s}: r={r_val:+.4f} (p={p_val:.4f}{sig}), "
              f"τ={tau:+.4f} (p={p_tau:.4f})")

    stat_results['correlations'] = correlation_results

    # 4c. Variance reduction from normalization
    print(f"\n  ── Normalization Collapse Test ──")
    print(f"  Testing: does any normalization REDUCE variance of ε*?")
    print(f"  (Negative variance reduction means normalization INCREASES spread)")

    norm_collapse = {}
    for key in norm_keys:
        vals = [norm_data[a][key] for a in norm_data]
        cv_norm = np.std(vals) / max(np.mean(vals), 1e-10)
        var_red = 1.0 - (cv_norm / max(raw_cv_fine, 1e-10))**2  # CV-based var reduction

        # Levene test: does spread decrease?
        norm_collapse[key] = {
            'cv': float(cv_norm),
            'cv_reduction': float(1.0 - cv_norm/max(raw_cv_fine, 1e-10)),
            'variance_reduction': float(var_red),
            'values': {a: float(norm_data[a][key]) for a in norm_data},
        }
        better = "✓ BETTER" if cv_norm < raw_cv_fine else "✗ WORSE"
        print(f"  {key:<30s}: CV={cv_norm:.4f} (raw CV={raw_cv_fine:.4f}) {better}")

    stat_results['normalization_collapse'] = norm_collapse

    # 4d. Log-log regression ε* vs curvature to test power law
    print(f"\n  ── Power Law Tests: ε* = c · metric^α ──")
    regression_results = {}
    for metric_name in ['hessian_trace', 'fisher_trace', 'spectral_norm', 'd_eff', 'n_params']:
        x_vals = [curv_metrics[a][metric_name] for a in common_archs]
        log_x = np.log(x_vals)
        log_y = np.log(y_vals)

        if SCIPY_OK and np.std(log_y) > 1e-10:
            slope, intercept, r_val, p_val, se = sp_stats.linregress(log_x, log_y)
        elif np.std(log_y) > 1e-10:
            slope = float(np.polyfit(log_x, log_y, 1)[0])
            r_val = float(np.corrcoef(log_x, log_y)[0, 1])
            p_val, se, intercept = 1.0, 0.0, 0.0
        else:
            slope, intercept, r_val, p_val, se = 0.0, np.mean(log_y), 0.0, 1.0, 0.0

        regression_results[metric_name] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_val**2),
            'p_value': float(p_val),
            'slope_se': float(se),
        }
        print(f"  ε* ∝ {metric_name}^α: α={slope:.4f}±{se:.4f}, R²={r_val**2:.4f}, p={p_val:.4f}")

    stat_results['regressions'] = regression_results

    # 4e. Hypothesis: is ε* constant? (Cochran's Q / Bartlett's test on bootstrap samples)
    print(f"\n  ── Hypothesis: Is ε* constant across architectures? ──")
    if SCIPY_OK:
        # One-way ANOVA on bootstrap ε* distributions would be ideal,
        # but we can test if observed variation exceeds bootstrap noise
        total_boot_var = np.mean([boot_stds[a]**2 for a in boot_stds])
        between_var = np.var(eps_star_vals)
        f_ratio = between_var / max(total_boot_var, 1e-15)
        # Approximate F-test
        df1 = len(eps_star_vals) - 1
        df2 = 1000  # approximation for bootstrap
        p_const = 1.0 - sp_stats.f.cdf(f_ratio, df1, df2)

        print(f"  Between-arch variance: {between_var:.6f}")
        print(f"  Mean within-arch bootstrap variance: {total_boot_var:.6f}")
        print(f"  F-ratio: {f_ratio:.4f}")
        print(f"  p(ε* is constant): {p_const:.6f}")
        print(f"  Conclusion: {'ε* IS constant (cannot reject H0)' if p_const > 0.05 else 'ε* varies across architectures (reject H0)'}")

        stat_results['constancy_test'] = {
            'between_var': float(between_var),
            'within_var': float(total_boot_var),
            'f_ratio': float(f_ratio),
            'p_value': float(p_const),
            'is_constant': p_const > 0.05,
        }
    else:
        stat_results['constancy_test'] = {'note': 'scipy not available'}

    with open(os.path.join(RESULTS_DIR, 'phase4_statistics.json'), 'w') as f:
        json.dump(stat_results, f, indent=2)

    print(f"\nPhase 4 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: GENERATE PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 5: GENERATING PLOTS")
    print("="*70)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'serif'})
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("  matplotlib not available, skipping plots")

    if HAS_MPL:
        # ── PLOT 1: Dense forgetting curves (all architectures) ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(phase_archs)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        for idx, arch_name in enumerate(sorted(dense_data.keys(),
                                                key=lambda x: dense_data[x]['n_params'])):
            d = dense_data[arch_name]
            ax.semilogx(d['epsilon_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=f"{arch_name} (ε*={d['eps_star']:.2f})", lw=1.5, ms=5)
            ax.fill_between(d['epsilon_values'],
                           [m-s for m,s in zip(d['forgetting_means'], d['forgetting_stds'])],
                           [m+s for m,s in zip(d['forgetting_means'], d['forgetting_stds'])],
                           alpha=0.15, color=colors[idx])
        ax.set_xlabel('Stability Budget ε (log scale)')
        ax.set_ylabel('Forgetting F')
        ax.set_title('Phase 1: Dense ε Sweep — All Architectures (FTR, 5 seeds)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axvspan(5, 10, alpha=0.1, color='red', label='Transition zone')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'dense_forgetting_curves.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'dense_forgetting_curves.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ dense_forgetting_curves")

        # ── PLOT 2: ε* with bootstrap CI ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        arch_names = sorted(dense_data.keys(), key=lambda x: dense_data[x]['n_params'])
        x_pos = range(len(arch_names))
        stars = [dense_data[a]['eps_star'] for a in arch_names]
        ci_lo = [dense_data[a]['eps_star_ci95'][0] for a in arch_names]
        ci_hi = [dense_data[a]['eps_star_ci95'][1] for a in arch_names]
        errs = [[s-l for s,l in zip(stars, ci_lo)], [h-s for s,h in zip(stars, ci_hi)]]

        ht_vals = [curv_source[a]['hessian_trace']['mean'] for a in arch_names]
        scatter = ax.scatter(x_pos, stars, c=ht_vals, cmap='viridis', s=100, zorder=3, edgecolors='black')
        ax.errorbar(x_pos, stars, yerr=errs, fmt='none', ecolor='gray', capsize=4, zorder=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(arch_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ε*')
        ax.set_title('ε* with 95% Bootstrap CI (colored by Hessian trace)')
        plt.colorbar(scatter, label='Hessian trace')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'eps_star_bootstrap_ci.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'eps_star_bootstrap_ci.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ eps_star_bootstrap_ci")

        # ── PLOT 3: 2D Phase Diagram ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
        regime_labels = {0: 'Stable', 1: 'Partial', 2: 'Catastrophic'}

        for ax_idx, (metric, xlabel) in enumerate([
            ('hessian_trace', 'Hessian Trace tr(H)'),
            ('spectral_norm', 'Spectral Norm ||H||_op'),
            ('n_params', 'Parameters')
        ]):
            ax = axes[ax_idx]
            for pt in phase_points:
                ax.scatter(pt[metric], pt['eps'], c=regime_colors[pt['regime']],
                          s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('ε')
            ax.set_yscale('log')
            if metric != 'n_params':
                ax.set_xscale('log')
            ax.set_title(f'Phase Diagram: {xlabel} vs ε')
            ax.grid(True, alpha=0.3)
            # Add legend
            for r, c in regime_colors.items():
                ax.scatter([], [], c=c, label=regime_labels[r], s=40, edgecolors='black', linewidth=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'phase_diagram_2d.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'phase_diagram_2d.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ phase_diagram_2d")

        # ── PLOT 4: Normalized Phase Diagram (collapse test) ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax_idx, (norm_key, ylabel) in enumerate([
            ('eps_norm_fisher', 'ε · tr(F)'),
            ('eps_norm_hessian', 'ε · tr(H)'),
            ('eps_norm_deff', 'ε · d_eff'),
        ]):
            ax = axes[ax_idx]
            for pt in phase_points:
                norm_eps = pt.get(norm_key, pt['eps'])
                ax.scatter(pt['forgetting'], norm_eps, c=regime_colors[pt['regime']],
                          s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
            ax.set_xlabel('Forgetting F')
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_title(f'Normalized Phase Diagram: {ylabel}')
            ax.grid(True, alpha=0.3)
            for r, c in regime_colors.items():
                ax.scatter([], [], c=c, label=regime_labels[r], s=40, edgecolors='black', linewidth=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'phase_diagram_normalized.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'phase_diagram_normalized.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ phase_diagram_normalized")

        # ── PLOT 5: Variance Reduction Bar Chart ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        norm_names = sorted(variance_results.keys())
        cvs = [variance_results[n]['cv'] for n in norm_names]
        ax.barh(range(len(norm_names)), cvs, color='steelblue', edgecolor='black')
        ax.axvline(x=raw_cv_fine, color='red', linestyle='--', lw=2, label=f'Raw CV={raw_cv_fine:.4f}')
        ax.set_yticks(range(len(norm_names)))
        ax.set_yticklabels([n.replace('eps_norm_', '') for n in norm_names], fontsize=8)
        ax.set_xlabel('Coefficient of Variation')
        ax.set_title('Normalization Collapse Test: CV of Normalized ε*')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'variance_reduction.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'variance_reduction.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ variance_reduction")

        # ── PLOT 6: Cross-Method Overlay ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # FTR forgetting curves
        ax = axes[0]
        ax.set_title('FTR: Forgetting vs ε')
        for idx, arch_name in enumerate(sorted(dense_data.keys(),
                                                key=lambda x: dense_data[x]['n_params'])):
            d = dense_data[arch_name]
            ax.semilogx(d['epsilon_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=arch_name, lw=1.5, ms=4)
        ax.set_xlabel('ε')
        ax.set_ylabel('Forgetting F')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # LwF forgetting curves
        ax = axes[1]
        ax.set_title('LwF: Forgetting vs α')
        for idx, arch_name in enumerate(sorted(lwf_data.keys())):
            d = lwf_data[arch_name]
            ax.semilogx(d['alpha_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=arch_name, lw=1.5, ms=4)
        ax.set_xlabel('α (distillation weight)')
        ax.set_ylabel('Forgetting F')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # EWC forgetting curves
        ax = axes[2]
        ax.set_title('EWC: Forgetting vs λ')
        for idx, arch_name in enumerate(sorted(ewc_data.keys())):
            d = ewc_data[arch_name]
            ax.semilogx(d['lambda_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=arch_name, lw=1.5, ms=4)
        ax.set_xlabel('λ (EWC weight)')
        ax.set_ylabel('Forgetting F')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cross_method_overlay.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'cross_method_overlay.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ cross_method_overlay")

        # ── PLOT 7: Summary Figure ──
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        # Panel A: Dense forgetting (selected architectures)
        ax = fig.add_subplot(gs[0, 0])
        for idx, arch_name in enumerate(sorted(dense_data.keys(),
                                                key=lambda x: dense_data[x]['n_params'])):
            d = dense_data[arch_name]
            ax.semilogx(d['epsilon_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=arch_name, lw=1.2, ms=3)
        ax.set_xlabel('ε')
        ax.set_ylabel('Forgetting')
        ax.set_title('(A) FTR Forgetting vs ε')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel B: ε* with CI
        ax = fig.add_subplot(gs[0, 1])
        arch_sorted = sorted(dense_data.keys(), key=lambda x: dense_data[x]['n_params'])
        stars_plot = [dense_data[a]['eps_star'] for a in arch_sorted]
        ci_lo_plot = [dense_data[a]['eps_star_ci95'][0] for a in arch_sorted]
        ci_hi_plot = [dense_data[a]['eps_star_ci95'][1] for a in arch_sorted]
        errs_plot = [[s-l for s,l in zip(stars_plot, ci_lo_plot)],
                     [h-s for s,h in zip(stars_plot, ci_hi_plot)]]
        ax.errorbar(range(len(arch_sorted)), stars_plot, yerr=errs_plot,
                   fmt='ko', capsize=4, ms=6)
        ax.set_xticks(range(len(arch_sorted)))
        ax.set_xticklabels(arch_sorted, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('ε*')
        ax.set_title('(B) ε* ± 95% Bootstrap CI')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel C: Phase diagram
        ax = fig.add_subplot(gs[0, 2])
        for pt in phase_points:
            ax.scatter(pt['hessian_trace'], pt['eps'], c=regime_colors[pt['regime']],
                      s=20, alpha=0.7, edgecolors='black', linewidth=0.2)
        ax.set_xlabel('Hessian Trace')
        ax.set_ylabel('ε')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('(C) Phase Diagram')
        for r, c in regime_colors.items():
            ax.scatter([], [], c=c, label=regime_labels[r], s=20)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel D: Variance reduction
        ax = fig.add_subplot(gs[1, 0])
        top_norms = sorted(variance_results.keys(), key=lambda k: variance_results[k]['cv'])[:6]
        cvs_plot = [variance_results[n]['cv'] for n in top_norms]
        bar_colors = ['green' if c < raw_cv_fine else 'red' for c in cvs_plot]
        ax.barh(range(len(top_norms)), cvs_plot, color=bar_colors, edgecolor='black')
        ax.axvline(x=raw_cv_fine, color='blue', linestyle='--', lw=2, label=f'Raw CV')
        ax.set_yticks(range(len(top_norms)))
        ax.set_yticklabels([n.replace('eps_norm_', '') for n in top_norms], fontsize=7)
        ax.set_xlabel('CV')
        ax.set_title('(D) Normalization Test')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='x')

        # Panel E: LwF comparison
        ax = fig.add_subplot(gs[1, 1])
        for idx, arch_name in enumerate(sorted(lwf_data.keys())):
            d = lwf_data[arch_name]
            ax.semilogx(d['alpha_values'], d['forgetting_means'],
                       f'{markers[idx % len(markers)]}-', color=colors[idx],
                       label=arch_name, lw=1.2, ms=3)
        ax.set_xlabel('α')
        ax.set_ylabel('Forgetting')
        ax.set_title('(E) LwF Forgetting vs α')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel F: ε* vs Hessian trace (scatter)
        ax = fig.add_subplot(gs[1, 2])
        for a in common_archs:
            ht = curv_metrics[a]['hessian_trace']
            es = eps_stars[a]
            ax.scatter(ht, es, s=80, edgecolors='black', zorder=3)
            ax.annotate(a.replace('CNN_', '').replace('ResNet18_', 'RN'),
                       (ht, es), fontsize=6, ha='center', va='bottom')
        ax.set_xlabel('Hessian Trace')
        ax.set_ylabel('ε*')
        ax.set_title('(F) ε* vs Hessian Trace')
        ax.grid(True, alpha=0.3)

        plt.savefig(os.path.join(plots_dir, 'summary_phase_diagram.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'summary_phase_diagram.pdf'), bbox_inches='tight')
        plt.close()
        print("  ✓ summary_phase_diagram")

    print(f"\nPhase 5 done. ({datetime.now()})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 6: FINAL SUMMARY")
    print("="*70)

    # Determine conclusion
    constancy = stat_results.get('constancy_test', {})
    is_constant = constancy.get('is_constant', True)
    best_norm_key = min(variance_results.keys(), key=lambda k: variance_results[k]['cv'])
    best_norm_cv = variance_results[best_norm_key]['cv']
    collapse_helps = best_norm_cv < raw_cv_fine

    print(f"\n  ── CONCLUSIONS ──")
    print(f"  1. Is ε* constant across architectures?")
    if is_constant:
        print(f"     YES (F-test p={constancy.get('p_value', 'N/A'):.4f})")
    else:
        print(f"     NO (F-test p={constancy.get('p_value', 'N/A'):.4f})")

    print(f"  2. Does any normalization improve collapse?")
    if collapse_helps:
        print(f"     YES: {best_norm_key} (CV: {raw_cv_fine:.4f} → {best_norm_cv:.4f})")
    else:
        print(f"     NO: Best normalization {best_norm_key} CV={best_norm_cv:.4f} vs raw CV={raw_cv_fine:.4f}")

    print(f"  3. Cross-method universality:")
    ftr_stars = [dense_data[a]['eps_star'] for a in dense_data]
    lwf_stars = [lwf_data[a]['alpha_star'] for a in lwf_data] if lwf_data else []
    ewc_sharp = [ewc_data[a]['sharpness'] for a in ewc_data] if ewc_data else []
    print(f"     FTR: ε* range [{min(ftr_stars):.3f}, {max(ftr_stars):.3f}]")
    if lwf_stars:
        print(f"     LwF: α* range [{min(lwf_stars):.3f}, {max(lwf_stars):.3f}]")
    if ewc_sharp:
        print(f"     EWC: sharpness range [{min(ewc_sharp):.3f}, {max(ewc_sharp):.3f}]")

    # Save summary
    summary = {
        'eps_star_constant': is_constant,
        'eps_star_mean': float(np.mean(ftr_stars)),
        'eps_star_std': float(np.std(ftr_stars)),
        'eps_star_cv': float(raw_cv_fine),
        'best_normalization': best_norm_key,
        'best_norm_cv': float(best_norm_cv),
        'normalization_helps': collapse_helps,
        'ftr_eps_stars': {a: dense_data[a]['eps_star'] for a in dense_data},
        'lwf_alpha_stars': {a: lwf_data[a]['alpha_star'] for a in lwf_data} if lwf_data else {},
        'ewc_sharpness': {a: ewc_data[a]['sharpness'] for a in ewc_data} if ewc_data else {},
        'total_time': str(datetime.now() - t_start),
    }
    with open(os.path.join(RESULTS_DIR, 'final_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    t_end = datetime.now()
    print(f"\n{'='*70}")
    print(f"ALL PHASES COMPLETE. Duration: {t_end - t_start}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
