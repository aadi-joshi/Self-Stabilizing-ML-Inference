#!/usr/bin/env python3
"""
Renders the FTR per-task training procedure (Algorithm 1 in the paper) as a
flowchart: warmup, the constrained dual-ascent inner loop (Eqs. 1-5), and
the lambda-reset event at every task boundary that Section 5 diagnoses as
a confound. Purely illustrative of control flow, not measured data.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5})

NAVY = '#2C3E5C'
ORANGE = '#DD8452'
GRAY = '#8C8C8C'
LGRAY = '#EDEDED'
LNAVY = '#E7ECF3'
LORANGE = '#FBEADD'

fig, ax = plt.subplots(figsize=(7.0, 9.4))
ax.set_xlim(-0.3, 11.0)
ax.axis('off')


def box(cx, cy, w, h, text, fc=LNAVY, ec=NAVY, fontsize=9.2, weight='normal', lw=1.3, zorder=3, textcolor='#1A1A1A'):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle='round,pad=0.02,rounding_size=0.16',
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
             color=textcolor, weight=weight, zorder=zorder + 1, linespacing=1.35)


def diamond(cx, cy, w, h, text, fc=LORANGE, ec=ORANGE, fontsize=9.0):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    d = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
             color='#1A1A1A', zorder=4, linespacing=1.3)


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.3, style='-|>', zorder=2):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                         color=color, linewidth=lw, zorder=zorder)
    ax.add_patch(a)


def tag(x, y, text, fontsize=8.3, color=NAVY, ha='left'):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va='center',
             style='italic', zorder=5)


# ---- flow, top to bottom -----------------------------------------------
cx = 4.6
y = 21.6
box(cx, y, 6.2, 0.85, r'Next task $T_t$ arrives', fc=LGRAY, ec=GRAY, weight='bold')

y2 = y - 1.35
arrow(cx, y - 0.425, cx, y2 + 0.5)
box(cx, y2, 6.6, 1.0,
    r'$\lambda \leftarrow \lambda_{\mathrm{init}}$,  $m \leftarrow 0$' + '\n'
    r'(dual variable reset at every task boundary)',
    fc=LORANGE, ec=ORANGE, weight='bold')
tag(0.85, y2, 'Eq. 2', color=ORANGE)

y3 = y2 - 1.35
arrow(cx, y2 - 0.5, cx, y3 + 0.45)
box(cx, y3, 6.2, 0.9, r'Freeze reference model:  $f_{\mathrm{old}} \leftarrow f_\theta$')

y4 = y3 - 1.35
arrow(cx, y3 - 0.45, cx, y4 + 0.45)
box(cx, y4, 6.6, 1.0,
    r'Warmup: 1 epoch of gradient steps on $\mathcal{L}_{\mathrm{task}}$ only' + '\n'
    r'($\lambda$ held at $\lambda_{\mathrm{init}}$, constraint not enforced)')

# --- inner constrained loop, drawn as an enclosing panel -----------------
loop_top = y4 - 1.75
loop_bot = loop_top - 5.55
arrow(cx, y4 - 0.5, cx, loop_top + 0.85)

panel_top = loop_top + 1.05
panel_bot = loop_bot - 0.85
panel = FancyBboxPatch((0.15, panel_bot), 9.25, panel_top - panel_bot,
                        boxstyle='round,pad=0.02,rounding_size=0.18',
                        facecolor='none', edgecolor=NAVY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
ax.add_patch(panel)
ax.text(4.6, panel_top - 0.28, r'constrained step, $n = 1,\ldots,N$', fontsize=8.6,
        color=NAVY, style='italic', ha='center', zorder=4)

ys1 = loop_top
box(cx, ys1, 6.7, 1.05,
    r'Sample batch; compute $\mathcal{L}_{\mathrm{task}}(\theta)$ and' + '\n'
    r'$D_{\mathrm{KL}}(f_\theta \| f_{\mathrm{old}})$ at temperature $T$',
    fc='white', ec=NAVY)
tag(8.0, ys1 - 0.38, 'Eq. 1')

ys2 = ys1 - 1.45
arrow(cx, ys1 - 0.525, cx, ys2 + 0.55)
box(cx, ys2, 6.7, 1.0,
    r'Primal step: $\theta \leftarrow \theta - \eta_\theta \nabla_\theta \mathcal{L}(\theta,\lambda)$',
    fc='white', ec=NAVY)
tag(8.0, ys2, 'Eq. 3')

ys3 = ys2 - 1.45
arrow(cx, ys2 - 0.5, cx, ys3 + 0.6)
box(cx, ys3, 6.7, 1.15,
    r'Dual ascent: $m \leftarrow \rho m + \eta_\lambda(D_{\mathrm{KL}}-\varepsilon)$' + '\n'
    r'$\lambda \leftarrow \max(0,\, \lambda+m)$',
    fc='white', ec=NAVY)
tag(8.0, ys3, 'Eqs. 4-5')

# decision diamond at loop_bot
arrow(cx, ys3 - 0.575, cx, loop_bot + 0.62)
diamond(cx, loop_bot, 4.4, 1.3, r'$n = N$?')

# loop-back arrow (No) along the right side, inside the panel, clear of the Eq tags
right_x = 9.15
arrow(cx + 2.2, loop_bot, right_x, loop_bot, style='-', color=NAVY, lw=1.2)
arrow(right_x, loop_bot, right_x, ys1, style='-', color=NAVY, lw=1.2)
arrow(right_x, ys1, cx + 3.35, ys1, color=NAVY, lw=1.2)
ax.text(cx + 2.55, loop_bot + 0.32, 'no', fontsize=8.3, color=NAVY,
        ha='left', va='center', style='italic')

# exit downward (Yes)
y5 = panel_bot - 1.3
arrow(cx, loop_bot - 0.65, cx, y5 + 0.45)
ax.text(cx + 0.45, (loop_bot - 0.65 + y5 + 0.45) / 2 + 0.05, 'yes', fontsize=8.3, color=ORANGE, style='italic')
box(cx, y5, 5.2, 0.85, r'Task $t$ complete', fc=LGRAY, ec=GRAY, weight='bold')

# loop back to top for next task
left_x = -0.05
arrow(cx - 2.6, y5, left_x, y5, style='-', color=GRAY, lw=1.2)
arrow(left_x, y5, left_x, y, style='-', color=GRAY, lw=1.2)
arrow(left_x, y, cx - 3.1, y, color=GRAY, lw=1.2)
ax.text(left_x - 0.22, (y5 + y) / 2, r'$t \leftarrow t+1$', fontsize=8.3, color=GRAY,
        rotation=90, ha='center', va='center', style='italic')

# terminal
y6 = y5 - 1.3
arrow(cx, y5 - 0.425, cx, y6 + 0.4)
box(cx, y6, 5.0, 0.8, r'All tasks done: return $\theta$', fc='#1A2E45', ec='#1A2E45',
    fontsize=9.2, weight='bold', textcolor='white')

ax.set_ylim(y6 - 0.65, y + 0.65)
plt.tight_layout(pad=0.25)
plt.savefig(os.path.join(FIGDIR, 'fig_methodology_flowchart.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig_methodology_flowchart.png'), dpi=300, bbox_inches='tight')
plt.close()
print('wrote fig_methodology_flowchart')
