#!/usr/bin/env python3
"""Generate publication figures for the FTR v2 paper from the analyzed JSON."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, 'data')
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'font.family': 'serif', 'axes.linewidth': 0.8})

sig = json.load(open(os.path.join(DATA, 'dense_sweep_summary.json')))

FAMILY_COLORS = {'cnn': '#4C72B0', 'resnet': '#DD8452', 'mlp': '#55A868', 'vit': '#C44E52', 'mixer': '#8172B2'}
FAMILY_LABEL = {'cnn': 'CNN', 'resnet': 'ResNet(+Lite)', 'mlp': 'MLP', 'vit': 'ViT', 'mixer': 'MLP-Mixer'}


def sigmoid(x, f_min, f_max, k, x0):
    return f_min + (f_max - f_min) / (1.0 + np.exp(-k * (x - x0)))


# ---- Figure 1: all 30 forgetting curves colored by family ----
fig, ax = plt.subplots(figsize=(7.5, 5.2))
seen_fam = set()
for name, r in sorted(sig.items(), key=lambda kv: kv[1]['n_params']):
    fam = r['family']
    eps_vals = sorted(float(e) for e in r['forgetting_means'].keys())
    fg_vals = [r['forgetting_means'][str(e) if e != int(e) else f'{e:.1f}'] if False else
               r['forgetting_means'][ [k for k in r['forgetting_means'] if float(k) == e][0] ]
               for e in eps_vals]
    color = FAMILY_COLORS[fam]
    label = FAMILY_LABEL[fam] if fam not in seen_fam else None
    seen_fam.add(fam)
    ax.semilogx(eps_vals, fg_vals, '-', color=color, alpha=0.55, lw=1.1, label=label)
ax.set_xlabel(r'Stability budget $\varepsilon$ (log scale)')
ax.set_ylabel('Forgetting $F$')
ax.set_title('Dense $\\varepsilon$ sweep, all 30 architectures, colored by family')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig1_all_forgetting_curves.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig1_all_forgetting_curves.png'), bbox_inches='tight')
plt.close()
print('wrote fig1_all_forgetting_curves')

# ---- Figure 2: eps* with 95% CI per architecture, colored/grouped by family, sorted ----
fig, ax = plt.subplots(figsize=(7.5, 5.5))
items = sorted(sig.items(), key=lambda kv: (kv[1]['family'], kv[1]['eps_star_sigmoid']))
names = [k for k, v in items]
stars = [v['eps_star_sigmoid'] for k, v in items]
los = [v['eps_star_sigmoid'] - v['boot_ci95'][0] for k, v in items]
his = [v['boot_ci95'][1] - v['eps_star_sigmoid'] for k, v in items]
colors = [FAMILY_COLORS[v['family']] for k, v in items]
y_pos = np.arange(len(names))
ax.errorbar(stars, y_pos, xerr=[los, his], fmt='none', ecolor='gray', capsize=2, zorder=1, lw=0.8)
ax.scatter(stars, y_pos, c=colors, s=22, zorder=2, edgecolors='black', linewidths=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=6)
ax.set_xlabel(r'$\varepsilon^*$ (sigmoid fit, 95% bootstrap CI)')
ax.set_title('Crossover $\\varepsilon^*$ across all 30 architectures')
ax.grid(True, alpha=0.3, axis='x')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7, label=FAMILY_LABEL[f])
           for f, c in FAMILY_COLORS.items()]
ax.legend(handles=handles, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig2_eps_star_by_family.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig2_eps_star_by_family.png'), bbox_inches='tight')
plt.close()
print('wrote fig2_eps_star_by_family')

# ---- Figure 3: diagnostic sensitivity, forest-plot style ----
diag = json.load(open(os.path.join(DATA, 'diagnostic_summary.json')))
diag_wide = json.load(open(os.path.join(DATA, 'diagnostic_wide_summary.json')))
COND_ORDER = ['lambda_init_div5', 'eta_lambda_x5', 'epochs_x2', 'momentum_low', 'temperature_div2',
              'baseline', 'temperature_x2', 'lambda_max_low', 'lambda_max_high',
              'epochs_div2', 'eta_lambda_div5', 'lambda_init_x5']
COND_LABEL = {
    'baseline': 'baseline', 'eta_lambda_x5': r'$\eta_\lambda \times 5$',
    'eta_lambda_div5': r'$\eta_\lambda \div 5$', 'lambda_init_x5': r'$\lambda_{init}\times 5$',
    'lambda_init_div5': r'$\lambda_{init}\div 5$', 'lambda_max_low': r'$\lambda_{max}=10$',
    'lambda_max_high': r'$\lambda_{max}=200$', 'epochs_x2': 'epochs$\\times2$',
    'epochs_div2': 'epochs$\\div2$', 'momentum_low': r'$\rho=0.5$',
    'temperature_x2': '$T\\times2$', 'temperature_div2': '$T\\div2$',
}
WIDE_CONDS = {'eta_lambda_div5', 'lambda_init_x5', 'lambda_init_div5', 'epochs_div2'}

fig, ax = plt.subplots(figsize=(7, 5.5))
y = np.arange(len(COND_ORDER))
for arch, dy, marker, color in [('CNN_W16', 0.12, 'o', '#4C72B0'), ('CNN_W32', -0.12, 's', '#DD8452')]:
    vals, los, his = [], [], []
    for cond in COND_ORDER:
        src = diag_wide[arch][cond] if cond in WIDE_CONDS else diag[arch][cond]
        es = src['eps_star_sigmoid'] if cond in WIDE_CONDS else src['eps_star_sigmoid']
        ci = src['boot_ci95']
        vals.append(es)
        los.append(max(es - ci[0], 0))
        his.append(max(ci[1] - es, 0))
    ax.errorbar(vals, y + dy, xerr=[los, his], fmt=marker, color=color, ecolor=color,
                capsize=2, ms=5, label=arch.replace('_', ' '), lw=1.0)
ax.set_yticks(y)
ax.set_yticklabels([COND_LABEL[c] for c in COND_ORDER])
ax.set_xscale('log')
ax.axvline(x=diag['CNN_W16']['baseline']['eps_star_sigmoid'], color='gray', ls='--', lw=0.8, alpha=0.6)
ax.set_xlabel(r'$\varepsilon^*$ (log scale)')
ax.set_title('Optimizer-schedule sensitivity of $\\varepsilon^*$\n(architecture and task held fixed)')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig3_diagnostic_forest.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig3_diagnostic_forest.png'), bbox_inches='tight')
plt.close()
print('wrote fig3_diagnostic_forest')

# ---- Figure 4: finite-size scaling (width family, k vs W) ----
fss = json.load(open(os.path.join(DATA, 'finite_size_scaling.json')))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
ax.loglog(fss['widths'], fss['sharpness_k'], 'o-', color='#4C72B0')
ax.set_xlabel('CNN width $W$')
ax.set_ylabel('Sigmoid sharpness $k$')
ax.set_title(f"$k \\sim W^\\alpha$, $\\alpha={fss['power_law_alpha']:.2f}\\pm{fss['power_law_alpha_se']:.2f}$, "
             f"$p={fss['p_value']:.2f}$ (n.s.)")
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.semilogx(fss['widths'], fss['eps_star'], 's-', color='#DD8452')
ax.set_xlabel('CNN width $W$')
ax.set_ylabel(r'$\varepsilon^*$')
ax.set_title(r'$\varepsilon^*$ vs.\ width (schedule fixed)')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig4_finite_size_scaling.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig4_finite_size_scaling.png'), bbox_inches='tight')
plt.close()
print('wrote fig4_finite_size_scaling')

print('\nAll figures written to', FIGDIR)
