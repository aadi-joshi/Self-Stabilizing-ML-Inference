#!/usr/bin/env python3
"""Data-driven graphical abstract, replacing the earlier AI-generated
illustration. Two panels built from the paper's own measured results:
(A) the schedule's effect on eps* dwarfs the architecture-range effect;
(B) holding S = lambda_init/(eta_lambda*N) fixed collapses that schedule
effect back down, per architecture."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')
FIGDIR = os.path.join(HERE, 'figures')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'serif', 'axes.linewidth': 0.8})

NAVY = '#2C3E5C'
ORANGE = '#DD8452'
GRAY = '#8C8C8C'
FAMILY_COLORS = {'cnn': '#4C72B0', 'resnet': '#DD8452', 'mlp': '#55A868', 'vit': '#C44E52', 'mixer': '#8172B2'}

fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.32)

# ---- Panel A: architecture range vs schedule range (log scale dumbbells) ----
axA = fig.add_subplot(gs[0])

arch_lo, arch_hi = 6.10, 21.45
sched_lo, sched_hi = 0.97, 90.13

rows = [
    ("Architecture\n(30 models, schedule fixed)", arch_lo, arch_hi, NAVY),
    ("Dual-ascent schedule\n(1 model, 6 perturbations)", sched_lo, sched_hi, ORANGE),
]
y = [1, 0]
for (label, lo, hi, color), yy in zip(rows, y):
    axA.plot([lo, hi], [yy, yy], color=color, lw=6, solid_capstyle='round', zorder=2)
    axA.scatter([lo, hi], [yy, yy], color=color, s=70, zorder=3, edgecolors='white', linewidths=1.2)
axA.set_yticks(y)
axA.set_yticklabels([r[0] for r in rows], fontsize=10)
axA.set_xscale('log')
axA.set_xlim(0.5, 150)
axA.set_xlabel(r'range of $\varepsilon^*$ (log scale)')
axA.set_title('The schedule moves $\\varepsilon^*$\nfarther than architecture does', fontsize=11)
axA.text(arch_hi * 1.15, 1, f'{arch_hi/arch_lo:.0f}$\\times$', va='center', fontsize=9, color=NAVY)
axA.text(sched_hi * 1.15, 0, f'{sched_hi/sched_lo:.0f}$\\times$', va='center', fontsize=9, color=ORANGE)
axA.spines[['top', 'right']].set_visible(False)
axA.grid(True, axis='x', alpha=0.25)
axA.set_ylim(-0.7, 1.7)

# ---- Panel B: S-invariance strip plot ----
axB = fig.add_subplot(gs[1])

d = json.load(open(os.path.join(DATA, 'round2_summary.json')))
si = d['s_invariance']
ARCHS = ['ViT_Tiny', 'MLP_H128', 'ResNet18_W16', 'CNN_W32', 'CNN_W16']
CONDS = ['baseline', 's_matched_scale_up', 's_matched_scale_down', 's_matched_via_steps_a', 's_matched_via_steps_b']
RELOCATED = {('MLP_H128', 'baseline'): 12.84, ('ResNet18_W16', 'baseline'): 11.26,
             ('ViT_Tiny', 'baseline'): 9.99, ('ViT_Tiny', 's_matched_scale_up'): 18.21}

for i, arch in enumerate(ARCHS):
    row = si[arch]
    matched = [RELOCATED.get((arch, c), row[c]['crossover']) for c in CONDS]
    mismatched = row['mismatched_control']['crossover']
    axB.scatter(matched, [i] * len(matched), color=NAVY, s=42, zorder=3,
                edgecolors='white', linewidths=0.8,
                label=r'$S=8.33$ (5 routes)' if i == 0 else None)
    axB.scatter([mismatched], [i], color=ORANGE, s=70, marker='X', zorder=3,
                edgecolors='white', linewidths=0.8,
                label=r'$S=1.67$ (mismatched)' if i == 0 else None)
    axB.plot([min(matched), max(matched)], [i, i], color=NAVY, lw=1.0, alpha=0.35, zorder=1)

axB.set_yticks(range(len(ARCHS)))
axB.set_yticklabels([a.replace('_', ' ') for a in ARCHS], fontsize=10)
axB.set_xscale('log')
axB.set_xlim(0.6, 40)
axB.set_ylim(-0.9, len(ARCHS) - 0.3)
axB.set_xlabel(r'$\varepsilon^*$ (log scale)')
axB.set_title('Holding $S=\\lambda_{init}/(\\eta_\\lambda N)$ fixed\ncollapses that spread back down', fontsize=11)
axB.legend(fontsize=8.5, loc='lower center', bbox_to_anchor=(0.5, -0.32),
           ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.4)
axB.spines[['top', 'right']].set_visible(False)
axB.grid(True, axis='x', alpha=0.25)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(FIGDIR, 'graphical_abstract.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'graphical_abstract.png'), bbox_inches='tight', dpi=300)
plt.close()
print('wrote graphical_abstract (pdf + png)')
