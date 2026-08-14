#!/usr/bin/env python3
"""Analyze cross_method, cifar100_granularity, task_orderings, kl_ablation,
class_incremental. Point-estimate sigmoid fits only (no bootstrap -- these
are exploratory/robustness checks, not headline numbers that need tight
CIs). Writes data/secondary_summary.json."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')


def load(name):
    return json.load(open(os.path.join(DATA, f'{name}_raw.json')))


def fit_group(eps_map):
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    return {'crossover': es, 'k': k, 'r2': r2,
            'fg_means': dict(zip([str(e) for e in eps_vals], fg_means))}


out = {}
t0 = time.time()

# ============ cross_method ============
print("=== cross_method ===", flush=True)
d = load('cross_method')
groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['method'], key['arch'])
    groups.setdefault(gk, {}).setdefault(key['hyper_value'], []).append(v['forgetting'])

cm_out = {}
for (method, arch), eps_map in groups.items():
    fit = fit_group(eps_map)
    cm_out.setdefault(method, {})[arch] = fit
out['cross_method'] = cm_out
print(f"  done in {time.time()-t0:.1f}s", flush=True)

for method in cm_out:
    vals = [v['crossover'] for v in cm_out[method].values()]
    cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else float('nan')
    print(f"  {method}: n={len(vals)} mean_crossover={np.mean(vals):.3f} cv={cv*100:.1f}%", flush=True)

# ============ cifar100_granularity ============
print("=== cifar100_granularity ===", flush=True)
t1 = time.time()
d = load('cifar100_granularity')
groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['arch'], key['cpt'])
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

gran_out = {}
for (arch, cpt), eps_map in groups.items():
    fit = fit_group(eps_map)
    gran_out.setdefault(arch, {})[str(cpt)] = fit
out['cifar100_granularity'] = gran_out
print(f"  done in {time.time()-t1:.1f}s", flush=True)

print(f"  {'arch':<14s} {'cpt=2':>8s} {'cpt=4':>8s} {'cpt=5':>8s} {'cpt=10':>8s}", flush=True)
for arch in gran_out:
    row = gran_out[arch]
    vals = {cpt: row[cpt]['crossover'] if cpt in row else float('nan') for cpt in ['2', '4', '5', '10']}
    print(f"  {arch:<14s} {vals['2']:>8.2f} {vals['4']:>8.2f} {vals['5']:>8.2f} {vals['10']:>8.2f}", flush=True)

xs, ys = [], []
for arch in gran_out:
    for cpt_str, fit in gran_out[arch].items():
        xs.append(int(cpt_str))
        ys.append(fit['crossover'])
from scipy import stats as sp_stats
r, p = sp_stats.pearsonr(xs, ys)
print(f"  correlation(crossover, classes_per_task) across all cells: r={r:.3f} p={p:.4f} n={len(xs)}", flush=True)
out['cifar100_granularity_correlation'] = {'r': float(r), 'p': float(p), 'n': len(xs)}

# ============ task_orderings ============
print("=== task_orderings ===", flush=True)
t1 = time.time()
d = load('task_orderings')
groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['arch'], key['perm_seed'])
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

order_out = {}
for (arch, perm), eps_map in groups.items():
    fit = fit_group(eps_map)
    order_out.setdefault(arch, {})[str(perm)] = fit
out['task_orderings'] = order_out
print(f"  done in {time.time()-t1:.1f}s", flush=True)

print(f"  {'arch':<14s} " + " ".join(f"{'perm'+str(p):>10s}" for p in [1001, 1002, 1003]) + f"{'CV%':>8s}", flush=True)
for arch in order_out:
    vals = [order_out[arch][str(p)]['crossover'] for p in [1001, 1002, 1003] if str(p) in order_out[arch]]
    cv = float(np.std(vals) / np.mean(vals) * 100) if vals else float('nan')
    print(f"  {arch:<14s} " + " ".join(f"{v:>10.2f}" for v in vals) + f"{cv:>8.1f}", flush=True)

# ============ kl_ablation ============
print("=== kl_ablation ===", flush=True)
t1 = time.time()
d = load('kl_ablation')
groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = (key['arch'], key['direction'])
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

kl_out = {}
for (arch, direction), eps_map in groups.items():
    fit = fit_group(eps_map)
    kl_out.setdefault(arch, {})[direction] = fit
out['kl_ablation'] = kl_out
print(f"  done in {time.time()-t1:.1f}s", flush=True)

print(f"  {'arch':<14s} {'forward':>10s} {'reverse':>10s} {'js':>10s}", flush=True)
for arch in kl_out:
    row = kl_out[arch]
    vals = {d_: row[d_]['crossover'] if d_ in row else float('nan') for d_ in ['forward', 'reverse', 'js']}
    print(f"  {arch:<14s} {vals['forward']:>10.2f} {vals['reverse']:>10.2f} {vals['js']:>10.2f}", flush=True)

# ============ class_incremental ============
print("=== class_incremental ===", flush=True)
t1 = time.time()
d = load('class_incremental')
groups = {}
for k, v in d.items():
    key = json.loads(k)
    gk = key['arch']
    groups.setdefault(gk, {}).setdefault(key['eps'], []).append(v['forgetting'])

ci_out = {}
for arch, eps_map in groups.items():
    fit = fit_group(eps_map)
    ci_out[arch] = fit
out['class_incremental'] = ci_out
print(f"  done in {time.time()-t1:.1f}s", flush=True)

dense = json.load(open(os.path.join(DATA, 'dense_sweep_summary.json')))
print(f"  {'arch':<14s} {'class-incr':>12s} {'task-incr':>12s} {'ratio':>8s}", flush=True)
for arch in ci_out:
    ci_val = ci_out[arch]['crossover']
    ti_val = dense[arch]['eps_star_sigmoid'] if arch in dense else float('nan')
    ratio = ci_val / ti_val if ti_val else float('nan')
    print(f"  {arch:<14s} {ci_val:>12.2f} {ti_val:>12.2f} {ratio:>8.2f}", flush=True)

with open(os.path.join(DATA, 'secondary_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nsaved secondary_summary.json, total time {time.time()-t0:.1f}s", flush=True)
