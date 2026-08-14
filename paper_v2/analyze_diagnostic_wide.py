#!/usr/bin/env python3
"""Analyze the wide-grid diagnostic follow-up: pins down eps* for the
conditions that saturated the original [1,20] grid."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis
import numpy as np

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, 'data', 'diagnostic_wide_raw.json')))

groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['arch'], key['cond'])
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

CONDS = ['baseline', 'eta_lambda_div5', 'lambda_init_x5', 'lambda_init_div5', 'epochs_div2']
ARCHS = ['CNN_W16', 'CNN_W32']

out = {}
for arch in ARCHS:
    out[arch] = {}
    for cond in CONDS:
        gk = (arch, cond)
        eps_map = groups[gk]
        eps_vals = sorted(eps_map.keys())
        fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
        fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
        es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
        boot_mean, boot_std, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(
            eps_vals, fg_per_seed, n_bootstrap=1000)
        frange = max(fg_means) - min(fg_means)
        out[arch][cond] = {
            'eps_star_sigmoid': es, 'k': k, 'r2': r2,
            'boot_mean': boot_mean, 'boot_std': boot_std, 'boot_ci95': [ci_lo, ci_hi],
            'forgetting_range': frange,
            'fg_means': dict(zip([str(e) for e in eps_vals], fg_means)),
        }

with open(os.path.join(HERE, 'data', 'diagnostic_wide_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f"{'arch':<10s} {'condition':<18s} {'eps*':>9s} {'boot_ci95':>20s} {'R2':>6s} {'frange':>7s}")
for arch in ARCHS:
    print(f'--- {arch} ---')
    for cond in CONDS:
        r = out[arch][cond]
        ci = f"[{r['boot_ci95'][0]:.2f},{r['boot_ci95'][1]:.2f}]"
        print(f"{arch:<10s} {cond:<18s} {r['eps_star_sigmoid']:>9.3f} {ci:>20s} {r['r2']:>6.3f} "
              f"{r['forgetting_range']:>7.3f}")
