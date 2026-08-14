#!/usr/bin/env python3
"""Figure 5: eps* by family, original (as-recorded) vs. corrected
(epoch-matched + wide-grid-relocated), side by side. Visualizes the
hierarchical-model correction of Section 7.2."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stability_constrained_selfimprovement'))
from campaign import analysis

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')
FIGDIR = os.path.join(HERE, 'figures')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'font.family': 'serif', 'axes.linewidth': 0.8})

FAMILY_COLORS = {'cnn': '#4C72B0', 'resnet': '#DD8452', 'mlp': '#55A868', 'vit': '#C44E52', 'mixer': '#8172B2'}
FAMILY_LABEL = {'cnn': 'CNN', 'resnet': 'ResNet(+Lite)', 'mlp': 'MLP', 'vit': 'ViT', 'mixer': 'MLP-Mixer'}
FAMILY_ORDER = ['mixer', 'resnet', 'vit', 'cnn', 'mlp']  # corrected-mean order

dense = json.load(open(os.path.join(DATA, 'dense_sweep_summary.json')))
hv2 = json.load(open(os.path.join(DATA, 'hierarchical_v2.json')))
matched = hv2['matched_fits']

EPOCH_MATCHED = ['ViT_Tiny', 'Mixer_Tiny', 'ViT_Small', 'Mixer_Small']
WIDE = ['ResNet18_W16', 'ResNetLite_W8_NoBN']

# original values
orig = {a: r['eps_star_sigmoid'] for a, r in dense.items()}
family_of = {a: r['family'] for a, r in dense.items()}

# corrected values
cw = json.load(open(os.path.join(DATA, 'crossover_wide_raw.json')))
cw_groups = {}
for k, v in cw.items():
    key = json.loads(k)
    cw_groups.setdefault(key['arch'], {}).setdefault(key['eps'], []).append(v['forgetting'])
wide_fits = {}
for arch in WIDE:
    eps_map = cw_groups[arch]
    eps_vals = sorted(eps_map.keys())
    fg_means = [float(np.mean(eps_map[e])) for e in eps_vals]
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    wide_fits[arch] = es

corrected = {}
for a in orig:
    if a in EPOCH_MATCHED:
        corrected[a] = matched[a]['eps_star_sigmoid']
    elif a in WIDE:
        corrected[a] = wide_fits[a]
    else:
        corrected[a] = orig[a]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)

for ax, values, title in [(axes[0], orig, 'As originally recorded'),
                           (axes[1], corrected, 'Corrected (epoch-matched + wide-grid)')]:
    fam_vals = {}
    for a, v in values.items():
        fam_vals.setdefault(family_of[a], []).append(v)
    y0 = 0
    yticks, yticklabels = [], []
    for fam in FAMILY_ORDER:
        vs = sorted(fam_vals[fam])
        n = len(vs)
        ys = np.arange(y0, y0 + n)
        ax.scatter(vs, ys, color=FAMILY_COLORS[fam], s=28, edgecolors='black', linewidths=0.3, zorder=2)
        mean_v = np.mean(vs)
        ax.plot([mean_v, mean_v], [y0 - 0.4, y0 + n - 1 + 0.4], color=FAMILY_COLORS[fam],
                lw=1.6, alpha=0.7, zorder=1)
        yticks.append(y0 + (n - 1) / 2.0)
        yticklabels.append(f"{FAMILY_LABEL[fam]}\n(n={n})")
        y0 += n + 1.2
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(r'$\varepsilon^*$')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 25)

fig.suptitle('Family clustering before and after correcting the two schedule/CI\n'
              'data-quality issues (Sections 13.2 and 6)', fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig5_family_correction.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig5_family_correction.png'), bbox_inches='tight')
plt.close()
print('wrote fig5_family_correction')
