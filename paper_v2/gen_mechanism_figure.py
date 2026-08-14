#!/usr/bin/env python3
"""
Simulates the FTR dual-ascent lambda dynamics (Eqs 3-5 of the paper) in
isolation, for a few epsilon values, to produce a mechanism figure showing
WHY the crossover exists: lambda stays engaged (tight eps) vs decays to
zero mid-task (loose eps) within the same fixed step budget N.

This is a pure simulation of the *update rule*, not a real training run
(no GPU needed): we model the drift rate d_bar(lambda) as a simple
decreasing function of lambda (more regularization suppresses drift),
calibrated so behavior qualitatively matches what Section 5 measured.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'serif', 'axes.linewidth': 0.8})

# Dual-ascent hyperparameters (Section 3/4 defaults)
lambda_init = 1.0
eta_lambda = 0.005
lambda_max = 50.0
beta = 0.9  # momentum
N = 24      # constrained steps per task (baseline)

d_unc = 5.0    # unconstrained drift rate (architecture/task property, illustrative)
d_min = 0.3    # residual drift rate at very high lambda (constraint never fully silences drift)


def drift_rate(lam):
    # Smooth, monotonically decreasing drift rate in lambda, saturating at d_min.
    return d_min + (d_unc - d_min) / (1.0 + lam)


def simulate(eps, N=N, lambda_init=lambda_init, eta_lambda=eta_lambda, beta=beta, lambda_max=lambda_max):
    lam = lambda_init
    v = 0.0
    lambdas, drifts = [lam], [drift_rate(lam)]
    for t in range(N):
        d = drift_rate(lam)
        viol = d - eps
        v = beta * v + (1 - beta) * viol
        lam = max(0.0, min(lambda_max, lam + eta_lambda * v))
        lambdas.append(lam)
        drifts.append(drift_rate(lam))
    return np.array(lambdas), np.array(drifts)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

eps_values = [2.0, 5.0, 8.33, 12.0, 20.0]
colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.9, len(eps_values)))

ax = axes[0]
for eps, c in zip(eps_values, colors):
    lambdas, _ = simulate(eps)
    ax.plot(range(len(lambdas)), lambdas, color=c, lw=1.8, label=f'$\\varepsilon={eps:g}$')
ax.set_xlabel('Constrained step $t$ (of $N=24$)')
ax.set_ylabel(r'Dual variable $\lambda_t$')
ax.set_title(r'(A) $\lambda$ trajectory: tight $\varepsilon$ stays engaged,'
             '\nloose $\\varepsilon$ decays to 0 mid-task', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.6)

ax = axes[1]
eps_fine = np.linspace(0.5, 25, 60)
final_lambdas = [simulate(e)[0][-1] for e in eps_fine]
ax.plot(eps_fine, final_lambdas, color='#4C72B0', lw=2)
# mark where this toy simulation's own lambda_end crosses zero (not the
# paper's real Eq. 8 estimate, which is a first-order approximation near
# lambda=0 and need not coincide exactly with this illustrative curve)
zero_idx = np.argmax(np.array(final_lambdas) <= 1e-6)
ax.axvline(eps_fine[zero_idx], color='#C44E52', ls='--', lw=1.3,
           label=r'$\lambda$ reaches 0 within budget $N$')
ax.set_xlabel(r'Stability budget $\varepsilon$')
ax.set_ylabel(r'$\lambda$ at end of task ($t=N$)')
ax.set_title('(B) Crossover: where $\\lambda$ transitions\nfrom engaged to fully decayed', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig0_mechanism.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGDIR, 'fig0_mechanism.png'), bbox_inches='tight')
plt.close()
print('wrote fig0_mechanism')
