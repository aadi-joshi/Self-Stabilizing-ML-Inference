#!/usr/bin/env python3
"""Analyze round-2 results: s_invariance, diagnostic_families,
cifar100_granularity_v2, crossover_wide, epoch_matched_control (ViT_Small/
Mixer_Small). Writes data/round2_summary.json. Run after pulling all round2
raw JSON files into paper_v2/data/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')


def load(name):
    path = os.path.join(DATA, f'{name}_raw.json')
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        return None
    return json.load(open(path))


def fit_group(eps_map, n_boot=300):
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    boot_mean, boot_std, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(eps_vals, fg_per_seed, n_bootstrap=n_boot)
    saturated = analysis.is_bound_saturated(es, eps_vals)
    return {'crossover': es, 'k': k, 'r2': r2, 'boot_ci95': [ci_lo, ci_hi],
            'saturated': saturated, 'fg_means': dict(zip([str(e) for e in eps_vals], fg_means))}


out = {}

# ============ s_invariance ============
print("=== s_invariance ===")
d = load('s_invariance')
if d:
    groups = {}
    for k, v in d.items():
        key = json.loads(k)
        gk = (key['arch'], key['cond'])
        groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])
    si_out = {}
    for (arch, cond), eps_map in groups.items():
        si_out.setdefault(arch, {})[cond] = fit_group(eps_map)
    out['s_invariance'] = si_out
    CONDS = ['baseline', 's_matched_scale_up', 's_matched_scale_down',
             's_matched_via_steps_a', 's_matched_via_steps_b', 'mismatched_control']
    print(f"  {'arch':<14s} " + " ".join(f"{c[:12]:>13s}" for c in CONDS))
    for arch in si_out:
        row = si_out[arch]
        vals = [row[c]['crossover'] if c in row else float('nan') for c in CONDS]
        print(f"  {arch:<14s} " + " ".join(f"{v:>13.2f}" for v in vals))
    # summary: CV across the 5 S-matched conditions (baseline + 4 matched) vs the mismatched control's deviation
    for arch in si_out:
        row = si_out[arch]
        matched = [row[c]['crossover'] for c in CONDS[:5] if c in row]
        cv_matched = float(np.std(matched) / np.mean(matched) * 100) if matched else float('nan')
        mismatched = row.get('mismatched_control', {}).get('crossover', float('nan'))
        ratio_mismatch = mismatched / np.mean(matched) if matched else float('nan')
        print(f"    {arch}: S-matched CV={cv_matched:.1f}%  mismatched_control/mean_matched={ratio_mismatch:.2f}")

# ============ diagnostic_families ============
print("\n=== diagnostic_families ===")
d = load('diagnostic_families')
if d:
    groups = {}
    for k, v in d.items():
        key = json.loads(k)
        gk = (key['arch'], key['cond'])
        groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])
    fam_out = {}
    for (arch, cond), eps_map in groups.items():
        fam_out.setdefault(arch, {})[cond] = fit_group(eps_map)
    out['diagnostic_families'] = fam_out
    CONDS = ['baseline', 'eta_lambda_x5', 'epochs_x2', 'lambda_init_x5', 'lambda_init_div5']
    print(f"  {'arch':<14s} " + " ".join(f"{c[:12]:>13s}" for c in CONDS))
    for arch in fam_out:
        row = fam_out[arch]
        vals = [row[c]['crossover'] if c in row else float('nan') for c in CONDS]
        sat = [row[c]['saturated'] if c in row else None for c in CONDS]
        print(f"  {arch:<14s} " + " ".join(f"{v:>13.2f}" for v in vals))
        print(f"    saturated: {sat}")

# ============ cifar100_granularity_v2 ============
print("\n=== cifar100_granularity_v2 ===")
d = load('cifar100_granularity_v2')
if d:
    groups = {}
    for k, v in d.items():
        key = json.loads(k)
        gk = (key['arch'], key['cpt'])
        groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])
    gran_out = {}
    for (arch, cpt), eps_map in groups.items():
        gran_out.setdefault(arch, {})[str(cpt)] = fit_group(eps_map)
    out['cifar100_granularity_v2'] = gran_out
    print(f"  {'arch':<14s} {'cpt=2':>8s} {'cpt=4':>8s} {'cpt=5':>8s} {'cpt=10':>8s}")
    for arch in gran_out:
        row = gran_out[arch]
        vals = {cpt: row[cpt]['crossover'] if cpt in row else float('nan') for cpt in ['2', '4', '5', '10']}
        print(f"  {arch:<14s} {vals['2']:>8.2f} {vals['4']:>8.2f} {vals['5']:>8.2f} {vals['10']:>8.2f}")
    from scipy import stats as sp_stats
    n_dec = 0
    n_tot = 0
    rhos = []
    for arch in gran_out:
        row = gran_out[arch]
        cpts = sorted(int(c) for c in row.keys())
        vals = [row[str(c)]['crossover'] for c in cpts]
        if len(vals) >= 3:
            rho, p = sp_stats.spearmanr(cpts, vals)
            rhos.append(rho)
            n_tot += 1
            if vals[0] > vals[-1]:
                n_dec += 1
    print(f"  {n_dec}/{n_tot} architectures monotonic decreasing; mean spearman rho={np.mean(rhos):.3f}")
    xs, ys = [], []
    for arch in gran_out:
        for cpt_str, fit in gran_out[arch].items():
            xs.append(int(cpt_str))
            ys.append(fit['crossover'])
    r, p = sp_stats.pearsonr(xs, ys)
    print(f"  pooled pearson r={r:.3f} p={p:.5f} n={len(xs)} (samples/task now held fixed at 800)")
    out['cifar100_granularity_v2_correlation'] = {'r': float(r), 'p': float(p), 'n': len(xs),
                                                    'mean_spearman_rho': float(np.mean(rhos)) if rhos else None}

# ============ crossover_wide ============
print("\n=== crossover_wide ===")
d = load('crossover_wide')
if d:
    groups = {}
    for k, v in d.items():
        key = json.loads(k)
        groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])
    wide_out = {}
    for arch, eps_map in groups.items():
        wide_out[arch] = fit_group(eps_map, n_boot=1000)
    out['crossover_wide'] = wide_out
    for arch, fit in wide_out.items():
        print(f"  {arch}: crossover={fit['crossover']:.2f} CI={fit['boot_ci95']} R2={fit['r2']:.3f} "
              f"saturated={fit['saturated']}")

# ============ epoch_matched_control (ViT_Small / Mixer_Small) ============
print("\n=== epoch_matched_control (ViT_Small/Mixer_Small) ===")
d = load('epoch_matched_control')
if d:
    groups = {}
    for k, v in d.items():
        key = json.loads(k)
        if key['arch'] not in ('ViT_Small', 'Mixer_Small'):
            continue
        groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])
    ctrl_out = {}
    for arch, eps_map in groups.items():
        ctrl_out[arch] = fit_group(eps_map, n_boot=1000)
    out['epoch_matched_control_v2'] = ctrl_out
    dense = json.load(open(os.path.join(DATA, 'dense_sweep_summary.json')))
    for arch, fit in ctrl_out.items():
        orig = dense[arch]['eps_star_sigmoid'] if arch in dense else None
        print(f"  {arch}: 4ep(matched)={fit['crossover']:.2f} CI={fit['boot_ci95']} R2={fit['r2']:.3f}  "
              f"vs original(5ep)={orig}")

with open(os.path.join(DATA, 'round2_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nsaved round2_summary.json")
