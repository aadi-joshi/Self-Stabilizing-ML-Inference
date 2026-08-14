#!/usr/bin/env python3
"""Recompute the hierarchical partial-pooling model, family table, and
leave-one-out sensitivity with two corrections applied to the 30-architecture
zoo: (1) epoch-matched values substituted for ViT_Tiny, Mixer_Tiny,
ViT_Small, Mixer_Small (all originally run at 5 epochs/task vs 4 for the
rest of the zoo -- see Section 13.2 self-confound), and (2) wide-grid
relocated values substituted for ResNet18_W16 and ResNetLite_W8_NoBN, whose
original bootstrap CIs pinned against the curve_fit search bound (see
crossover_wide follow-up, Section 6). Writes data/hierarchical_v2.json."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis, models as models_mod
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')

dense = json.load(open(os.path.join(DATA, 'dense_sweep_summary.json')))
zoo = models_mod.get_architecture_zoo()

EPOCH_MATCHED_ARCHS = ['ViT_Tiny', 'Mixer_Tiny', 'ViT_Small', 'Mixer_Small']

# refit the four epoch-matched architectures at n_bootstrap=1000 to match
# the main dense_sweep pipeline's precision
emc = json.load(open(os.path.join(DATA, 'epoch_matched_control_raw.json')))
groups = {}
for k, v in emc.items():
    key = json.loads(k)
    groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])

matched_fits = {}
for arch in EPOCH_MATCHED_ARCHS:
    eps_map = groups[arch]
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    boot_mean, boot_std, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(
        eps_vals, fg_per_seed, n_bootstrap=1000)
    sat = analysis.is_bound_saturated(es, eps_vals)
    matched_fits[arch] = {'eps_star_sigmoid': es, 'boot_std': boot_std,
                           'boot_ci95': [ci_lo, ci_hi], 'r2': r2, 'saturated': sat}
    print(f"{arch} (epoch-matched, 4ep): eps*={es:.3f} boot_std={boot_std:.3f} "
          f"CI={[round(ci_lo,2), round(ci_hi,2)]} R2={r2:.3f} sat={sat}")
    print(f"  (original 5ep value: {dense[arch]['eps_star_sigmoid']:.3f})")

# crossover_wide relocated values (wide-grid rerun for the two architectures
# whose original bootstrap CI pinned against the search bound)
r2s = json.load(open(os.path.join(DATA, 'round2_summary.json')))
WIDE_ARCHS = ['ResNet18_W16', 'ResNetLite_W8_NoBN']
wide_fits = {}
cw = json.load(open(os.path.join(DATA, 'crossover_wide_raw.json')))
cw_groups = {}
for k, v in cw.items():
    key = json.loads(k)
    cw_groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])
for arch in WIDE_ARCHS:
    eps_map = cw_groups[arch]
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    fg_per_seed = {str(e): eps_map[e] for e in eps_vals}
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    boot_mean, boot_std_, ci_lo, ci_hi = analysis.bootstrap_eps_star_sigmoid(
        eps_vals, fg_per_seed, n_bootstrap=1000)
    wide_fits[arch] = {'eps_star_sigmoid': es, 'boot_std': boot_std_,
                        'boot_ci95': [ci_lo, ci_hi], 'r2': r2}
    print(f"{arch} (wide-grid relocated): eps*={es:.3f} boot_std={boot_std_:.3f} "
          f"CI={[round(ci_lo,2), round(ci_hi,2)]} R2={r2:.3f}")
    print(f"  (original narrow-grid value: {dense[arch]['eps_star_sigmoid']:.3f}, "
          f"CI={dense[arch]['boot_ci95']})")

# build corrected eps_star / boot_std dicts: substitute the 4 epoch-matched
# architectures and the 2 wide-grid-relocated architectures, keep the other
# 24 as originally recorded
eps_star = {}
boot_std = {}
family_of = {}
for arch, r in dense.items():
    family_of[arch] = r['family']
    if arch in EPOCH_MATCHED_ARCHS:
        eps_star[arch] = matched_fits[arch]['eps_star_sigmoid']
        boot_std[arch] = matched_fits[arch]['boot_std']
    elif arch in WIDE_ARCHS:
        eps_star[arch] = wide_fits[arch]['eps_star_sigmoid']
        boot_std[arch] = wide_fits[arch]['boot_std']
    else:
        eps_star[arch] = r['eps_star_sigmoid']
        boot_std[arch] = r['boot_std']

vals = list(eps_star.values())
cv = float(np.std(vals) / np.mean(vals))
print(f"\nn={len(vals)} architectures (epoch-matched), mean eps* = {np.mean(vals):.3f}, "
      f"std = {np.std(vals):.3f}, CV = {cv*100:.2f}%, range=[{min(vals):.2f},{max(vals):.2f}]")

print("\n=== Hierarchical partial-pooling (epoch-matched) ===")
hb = analysis.hierarchical_partial_pooling(eps_star, boot_std, n_mcmc=30000, burn=6000, seed=1)
print(f"mu = {hb['mu_posterior_mean']:.3f} +/- {hb['mu_posterior_sd']:.3f}, 95% CI {hb['mu_ci95']}")
print(f"tau = {hb['tau_posterior_mean']:.3f}, 95% CI {hb['tau_ci95']}")
print(f"ICC = {hb['icc_posterior_mean']:.3f}, 95% CI {hb['icc_ci95']}")

print("\n=== Leave-one-out (top 5 most CV-changing drops, epoch-matched) ===")
loo = analysis.leave_one_out(eps_star, boot_std)
full_cv = loo['full']['cv']
deltas = sorted(((k, v['cv'] - full_cv) for k, v in loo.items() if k != 'full'),
                 key=lambda kv: abs(kv[1]), reverse=True)
for k, delta in deltas[:5]:
    print(f"  {k}: cv={loo[k]['cv']*100:.2f}% (full={full_cv*100:.2f}%, delta={delta*100:+.2f}pp)")

print("\n=== Family table (epoch-matched) ===")
fam_vals = {}
for arch, e in eps_star.items():
    fam_vals.setdefault(family_of[arch], []).append(e)
for fam, vs in sorted(fam_vals.items(), key=lambda kv: np.mean(kv[1])):
    print(f"  {fam:<10s} n={len(vs):<3d} mean={np.mean(vs):.2f} std={np.std(vs):.2f}")

with open(os.path.join(DATA, 'hierarchical_v2.json'), 'w') as f:
    json.dump({
        'matched_fits': matched_fits,
        'hierarchical': hb,
        'leave_one_out': {kk: vv for kk, vv in loo.items()},
        'raw_cv': cv, 'n_architectures': len(vals),
        'mean_eps_star': float(np.mean(vals)), 'std_eps_star': float(np.std(vals)),
        'family_table': {fam: {'n': len(vs), 'mean': float(np.mean(vs)), 'std': float(np.std(vs))}
                          for fam, vs in fam_vals.items()},
    }, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print("\nsaved data/hierarchical_v2.json")
