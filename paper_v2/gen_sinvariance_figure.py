#!/usr/bin/env python3
"""Figure 6: S-invariance. For each of 5 architectures, plot the 5
S-matched (S=8.33) crossover values as a cluster and the mismatched-control
(S=1.67) crossover as a distinct marker, log scale, to show the
'five-cluster, one-outlier' pattern that motivates Section 9."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')
FIGDIR = os.path.join(HERE, 'figures')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'font.family': 'serif', 'axes.linewidth': 0.8})

ARCHS = ['CNN_W16', 'CNN_W32', 'ResNet18_W16', 'MLP_H128', 'ViT_Tiny']
CONDS = ['baseline', 's_matched_scale_up', 's_matched_scale_down', 's_matched_via_steps_a', 's_matched_via_steps_b']

# relocated (wide-grid) values for the 4 originally-saturated cells
RELOCATED = {
    ('MLP_H128', 'baseline'): 12.84,
    ('ResNet18_W16', 'baseline'): 11.26,
    ('ViT_Tiny', 'baseline'): 9.99,
    ('ViT_Tiny', 's_matched_scale_up'): 18.21,
}

d = json.load(open(os.path.join(DATA, 'round2_summary.json')))
si = d['s_invariance']

fig, ax = plt.subplots(figsize=(7.2, 4.5))
y_pos = np.arange(len(ARCHS))

for i, arch in enumerate(ARCHS):
    row = si[arch]
    matched_vals = []
    for c in CONDS:
        v = RELOCATED.get((arch, c), row[c]['crossover'] if c in row else None)
        if v is not None:
            matched_vals.append(v)
    mismatched_val = row['mismatched_control']['crossover']
    jitter = np.random.RandomState(i).uniform(-0.12, 0.12, size=len(matched_vals))
    ax.scatter(matched_vals, [i + j for j in jitter], color='#4C72B0', s=45,
               edgecolors='black', linewidths=0.4, zorder=3,
               label='$S=8.33$ (5 routes)' if i == 0 else None)
    ax.scatter([mismatched_val], [i], color='#C44E52', s=70, marker='X',
               edgecolors='black', linewidths=0.5, zorder=3,
               label='$S=1.67$ (mismatched control)' if i == 0 else None)
    mean_matched = np.mean(matched_vals)
    ax.plot([mean_matched, mean_matched], [i - 0.3, i + 0.3], color='#4C72B0', lw=1.2, alpha=0.5, zorder=2)

ax.set_yticks(y_pos)
ax.set_yticklabels([a.replace('_', ' ') for a in ARCHS])
ax.set_xscale('log')
ax.set_xlabel(r'$\varepsilon^*$ (log scale)')
ax.set_title('Five routes to the same schedule ratio $S$ cluster;\na $5\\times$ change in $S$ itself does not')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig6_sinvariance.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig6_sinvariance.png'), bbox_inches='tight')
plt.close()
print('wrote fig6_sinvariance')
