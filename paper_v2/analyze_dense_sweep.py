#!/usr/bin/env python3
"""Analyze the 30-architecture main FTR eps-sweep on CIFAR-10.
Produces paper_v2/data/dense_sweep_summary.json used by Secs. 6-8."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis, models as models_mod
import numpy as np

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, 'data', 'dense_sweep_raw.json')))
curv_raw = json.load(open(os.path.join(HERE, 'data', 'curvature_full_zoo.json')))
zoo = models_mod.get_architecture_zoo()

# curv_raw is keyed by json-encoded {arch,seed,stage}; aggregate to per-architecture means
_curv_by_arch = {}
for k, v in curv_raw.items():
    if '_error' in v:
        continue
    key = json.loads(k)
    _curv_by_arch.setdefault(key['arch'], []).append(v)
curv = {}
for arch, entries in _curv_by_arch.items():
    curv[arch] = {
        m: float(np.mean([e[m] for e in entries]))
        for m in ['hessian_trace', 'fisher_trace', 'spectral_norm', 'd_eff', 'gradient_norm']
    }
    curv[arch]['n_params'] = entries[0]['n_params']
    curv[arch]['n_seeds'] = len(entries)

groups = {}
for k, v in d.items():
    key = json.loads(k)
    groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])
    if '_new_task_accuracy' not in v:
        pass

new_task_groups = {}
for k, v in d.items():
    key = json.loads(k)
    new_task_groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v.get('new_task_accuracy'))

print(f"Architectures with data: {len(groups)}")

sigmoid_results = {}
for arch, eps_map in groups.items():
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    boot_mean, boot_std, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(
        eps_vals, fg_per_seed, n_bootstrap=1000)
    n_seeds = len(list(eps_map.values())[0])
    bound_saturated = analysis.is_bound_saturated(es, eps_vals)
    sigmoid_results[arch] = {
        'eps_star_sigmoid': es, 'sigmoid_k': k, 'sigmoid_f_min': fmin, 'sigmoid_f_max': fmax,
        'sigmoid_r_sq': r2, 'boot_mean': boot_mean, 'boot_std': boot_std, 'boot_ci95': [ci_lo, ci_hi],
        'n_params': zoo[arch]['n_params'], 'family': zoo[arch]['family'], 'group': zoo[arch]['group'],
        'n_seeds': n_seeds, 'bound_saturated': bound_saturated,
        'forgetting_means': dict(zip([str(e) for e in eps_vals], fg_means)),
        'new_task_acc_means': {str(e): float(np.mean(new_task_groups[arch][e])) for e in eps_vals},
    }
    if bound_saturated:
        print(f"  WARNING: {arch} eps*={es:.3f} is pinned against the curve_fit search bound "
              f"(not a located transition) -- see crossover_wide follow-up")

with open(os.path.join(HERE, 'data', 'dense_sweep_summary.json'), 'w') as f:
    json.dump(sigmoid_results, f, indent=2)

print(f"\n{'arch':<22s} {'family':<8s} {'params':>10s} {'eps*':>8s} {'boot_ci95':>16s} {'R2':>6s} {'k':>6s}")
for arch in sorted(sigmoid_results, key=lambda a: sigmoid_results[a]['n_params']):
    r = sigmoid_results[arch]
    ci = f"[{r['boot_ci95'][0]:.2f},{r['boot_ci95'][1]:.2f}]"
    print(f"{arch:<22s} {r['family']:<8s} {r['n_params']:>10,d} {r['eps_star_sigmoid']:>8.3f} "
          f"{ci:>16s} {r['sigmoid_r_sq']:>6.3f} {r['sigmoid_k']:>6.2f}")

# ---- universality stats ----
eps_star = {a: v['eps_star_sigmoid'] for a, v in sigmoid_results.items()}
boot_std = {a: v['boot_std'] for a, v in sigmoid_results.items()}
vals = list(eps_star.values())
cv = float(np.std(vals) / np.mean(vals))
print(f"\nn={len(vals)} architectures, mean eps* = {np.mean(vals):.3f}, "
      f"std = {np.std(vals):.3f}, CV = {cv*100:.2f}%, range=[{min(vals):.2f},{max(vals):.2f}]")

print("\n=== Hierarchical partial-pooling ===")
hb = analysis.hierarchical_partial_pooling(eps_star, boot_std, n_mcmc=30000, burn=6000, seed=1)
print(f"mu = {hb['mu_posterior_mean']:.3f} +/- {hb['mu_posterior_sd']:.3f}, 95% CI {hb['mu_ci95']}")
print(f"tau = {hb['tau_posterior_mean']:.3f}, 95% CI {hb['tau_ci95']}")
print(f"ICC = {hb['icc_posterior_mean']:.3f}, 95% CI {hb['icc_ci95']}")

print("\n=== Leave-one-out (top 5 most CV-changing drops) ===")
loo = analysis.leave_one_out(eps_star, boot_std)
full_cv = loo['full']['cv']
deltas = sorted(((k, v['cv'] - full_cv) for k, v in loo.items() if k != 'full'),
                 key=lambda kv: abs(kv[1]), reverse=True)
for k, delta in deltas[:5]:
    print(f"  {k}: cv={loo[k]['cv']*100:.2f}% (full={full_cv*100:.2f}%, delta={delta*100:+.2f}pp)")

print("\n=== Correlation power + Bayes factors (n={}) ===".format(len(vals)))
cm = {}
for a in eps_star:
    c = curv.get(a)
    if not c:
        continue
    cm[a] = {
        'hessian_trace': c['hessian_trace'], 'fisher_trace': c['fisher_trace'],
        'spectral_norm': c['spectral_norm'], 'd_eff': c['d_eff'],
        'n_params': c['n_params'], 'gradient_norm': c['gradient_norm'],
    }
cp = analysis.correlation_power(eps_star, cm)
print(f"min detectable |r| at n={cp['n_architectures']}, 80% power: {cp['min_detectable_r_at_80pct_power']:.3f}")
for m, v in cp['correlations'].items():
    print(f"  {m:<16s} r={v['pearson_r']:+.3f} p={v['pearson_p']:.4f} BF10={v['bf10']:.4g}")

with open(os.path.join(HERE, 'data', 'dense_sweep_stats.json'), 'w') as f:
    json.dump({'hierarchical': hb, 'leave_one_out': loo, 'correlation_power': cp,
                'raw_cv': cv, 'n_architectures': len(vals),
                'mean_eps_star': float(np.mean(vals)), 'std_eps_star': float(np.std(vals))},
               f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

print("\n=== Finite-size scaling (CNN width family) ===")
width_archs = ['CNN_W8', 'CNN_W16', 'CNN_W24', 'CNN_W32', 'CNN_W48', 'CNN_W64', 'CNN_W96', 'CNN_W128']
widths = [8, 16, 24, 32, 48, 64, 96, 128]
present = [(a, w) for a, w in zip(width_archs, widths) if a in sigmoid_results]
fss = analysis.finite_size_scaling(sigmoid_results, [a for a, w in present], [w for a, w in present])
print(json.dumps(fss, indent=2))
with open(os.path.join(HERE, 'data', 'finite_size_scaling.json'), 'w') as f:
    json.dump(fss, f, indent=2)
