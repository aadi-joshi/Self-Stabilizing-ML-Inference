#!/usr/bin/env python3
"""Analyze the optimizer-schedule diagnostic sweep. Produces
paper_v2/data/diagnostic_summary.json used directly by Sec. 5 of the paper."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis
import numpy as np

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, 'data', 'diagnostic_raw.json')))

DIAG_EPS_GRID = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 20.0]

groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['arch'], key['cond'])
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

CONDS = ['baseline', 'eta_lambda_x5', 'eta_lambda_div5', 'lambda_init_x5', 'lambda_init_div5',
         'lambda_max_low', 'lambda_max_high', 'epochs_x2', 'epochs_div2', 'momentum_low',
         'temperature_x2', 'temperature_div2']
ARCHS = ['CNN_W16', 'CNN_W32']

out = {}
for arch in ARCHS:
    out[arch] = {}
    for cond in CONDS:
        gk = (arch, cond)
        if gk not in groups:
            continue
        eps_map = groups[gk]
        eps_vals = sorted(eps_map.keys())
        fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
        fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
        es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
        boot_mean, boot_std, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(
            eps_vals, fg_per_seed, n_bootstrap=1000)
        extrapolated = bool(es > max(eps_vals) or es < min(eps_vals))

        # Saturation check: a flat curve across the WHOLE tested grid means the
        # true crossover lies off-grid; the sigmoid fit degenerates to noise in
        # this case and its point estimate must not be reported as if it were a
        # located transition. mean-based, robust to a handful of noisy points.
        frange = max(fg_means) - min(fg_means)
        saturation = None
        if frange < 0.05:
            saturation = 'below_grid' if float(np.mean(fg_means)) > 0.12 else 'above_grid'

        out[arch][cond] = {
            'eps_star_sigmoid': es, 'k': k, 'r2': r2,
            'boot_mean': boot_mean, 'boot_std': boot_std, 'boot_ci95': [ci_lo, ci_hi],
            'extrapolated_beyond_grid': extrapolated,
            'forgetting_range': frange,
            'saturation': saturation,
            'reliable': bool(r2 >= 0.9 and not extrapolated and saturation is None),
            'fg_means': dict(zip([str(e) for e in eps_vals], fg_means)),
        }

# ratio vs baseline, for reliable fits only
for arch in ARCHS:
    base = out[arch]['baseline']['eps_star_sigmoid']
    for cond in CONDS:
        if cond in out[arch]:
            out[arch][cond]['ratio_vs_baseline'] = out[arch][cond]['eps_star_sigmoid'] / base

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
with open(os.path.join(HERE, 'data', 'diagnostic_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f"{'arch':<10s} {'condition':<20s} {'eps*':>8s} {'boot_ci95':>18s} {'R2':>6s} {'ratio':>7s} "
      f"{'reliable':>9s} {'saturation':>12s}")
for arch in ARCHS:
    print(f'--- {arch} ---')
    for cond in CONDS:
        if cond not in out[arch]:
            continue
        r = out[arch][cond]
        ci = f"[{r['boot_ci95'][0]:.2f},{r['boot_ci95'][1]:.2f}]"
        sat = r['saturation'] or '-'
        print(f"{arch:<10s} {cond:<20s} {r['eps_star_sigmoid']:>8.3f} {ci:>18s} {r['r2']:>6.3f} "
              f"{r['ratio_vs_baseline']:>7.2f} {str(r['reliable']):>9s} {sat:>12s}")
