#!/usr/bin/env python3
"""
Post-processing analysis for phase diagram experiments.
Reads raw data from phase_diagram results and applies
robust sigmoid-based ε* estimation + full normalization analysis.
"""

import os, sys, json, math
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'phase_diagram')
PREV_RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'neurips_breakthrough')


def sigmoid_fit_eps_star(eps_values, forgetting_values):
    """
    Fit logistic sigmoid to forgetting curve:
      F(ε) = F_min + (F_max - F_min) / (1 + exp(-k*(log(ε) - log(ε*))))
    Return (eps_star, sharpness_k, F_min, F_max, r_squared).
    """
    from scipy.optimize import curve_fit

    eps = np.array(eps_values, dtype=float)
    fg = np.array(forgetting_values, dtype=float)

    order = np.argsort(eps)
    eps = eps[order]
    fg = fg[order]
    log_eps = np.log(eps)

    def logistic(x, f_min, f_max, k, x0):
        return f_min + (f_max - f_min) / (1.0 + np.exp(-k * (x - x0)))

    # Initial guesses
    f_min0 = np.min(fg)
    f_max0 = np.max(fg)
    x0_0 = np.median(log_eps)
    k0 = 1.0

    try:
        popt, pcov = curve_fit(
            logistic, log_eps, fg,
            p0=[f_min0, f_max0, k0, x0_0],
            bounds=([0, 0, 0.01, log_eps[0] - 1], [1, 1, 50, log_eps[-1] + 1]),
            maxfev=10000
        )
        f_min, f_max, k, log_eps_star = popt
        eps_star = float(np.exp(log_eps_star))

        # R² of fit
        fg_pred = logistic(log_eps, *popt)
        ss_res = np.sum((fg - fg_pred) ** 2)
        ss_tot = np.sum((fg - np.mean(fg)) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

        return eps_star, float(k), float(f_min), float(f_max), float(r_sq)
    except Exception as e:
        print(f"    Sigmoid fit failed: {e}")
        return fallback_finite_diff(eps, fg)


def fallback_finite_diff(eps_sorted, fg_sorted):
    """Finite difference ε* estimation with interior-only search."""
    log_eps = np.log(eps_sorted)
    derivs = []
    for i in range(1, len(log_eps)):
        d_fg = fg_sorted[i] - fg_sorted[i-1]
        d_le = log_eps[i] - log_eps[i-1]
        derivs.append(abs(d_fg / d_le) if abs(d_le) > 1e-10 else 0.0)

    # Exclude boundary points (first and last) to avoid artifacts
    interior = derivs[1:-1] if len(derivs) > 3 else derivs
    offset = 1 if len(derivs) > 3 else 0
    max_idx = int(np.argmax(interior)) + offset
    eps_star = float(math.sqrt(eps_sorted[max_idx] * eps_sorted[max_idx + 1]))
    return eps_star, float(max(derivs)), 0.0, 0.0, 0.0


def bootstrap_eps_star_sigmoid(eps_values, forgetting_per_seed, n_bootstrap=2000):
    """Bootstrap ε* using sigmoid fit."""
    rng = np.random.RandomState(42)
    eps_arr = np.array(eps_values)

    n_seeds = len(list(forgetting_per_seed.values())[0])
    boot_stars = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_seeds, size=n_seeds)
        fg_means = []
        for eps in eps_values:
            vals = forgetting_per_seed[str(eps)]
            boot_vals = [vals[i] for i in idx if i < len(vals)]
            fg_means.append(float(np.mean(boot_vals)) if boot_vals else float(np.mean(vals)))

        try:
            es, _, _, _, _ = sigmoid_fit_eps_star(eps_values, fg_means)
            if 0.05 < es < 200:  # sanity check
                boot_stars.append(es)
        except Exception:
            pass

    if len(boot_stars) < 10:
        return np.mean(eps_values), np.std(eps_values), 0.0, 100.0

    bs = np.array(boot_stars)
    return float(np.mean(bs)), float(np.std(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    try:
        from scipy import stats as sp_stats
    except ImportError:
        sp_stats = None
        print("WARNING: scipy not available")

    # Load dense sweep data
    dense_path = os.path.join(RESULTS_DIR, 'phase1_dense_sweep.json')
    if not os.path.exists(dense_path):
        print(f"ERROR: {dense_path} not found. Run experiments first.")
        return

    with open(dense_path) as f:
        dense_data = json.load(f)

    print(f"Loaded dense sweep: {list(dense_data.keys())}")

    # Load curvature data — merge 5-seed (preferred) with original 3-seed fallback
    curv5_path = os.path.join(RESULTS_DIR, 'curvature_5seed.json')
    curv_orig = os.path.join(PREV_RESULTS, 'block_a_curvature.json')

    curv_data = {}
    if os.path.exists(curv_orig):
        with open(curv_orig) as f:
            curv_data = json.load(f)
        print(f"Loaded original 3-seed curvature: {sorted(curv_data.keys())}")

    if os.path.exists(curv5_path):
        with open(curv5_path) as f:
            curv5 = json.load(f)
        # Overwrite with higher-quality 5-seed data where available
        for arch, vals in curv5.items():
            curv_data[arch] = vals
        print(f"Merged 5-seed curvature for: {sorted(curv5.keys())}")

    if not curv_data:
        print("ERROR: No curvature data found")
        return

    # Show which sweep architectures have curvature
    sweep_archs = list(dense_data.keys())
    matched = [a for a in sweep_archs if a in curv_data]
    print(f"Curvature available for {len(matched)}/{len(sweep_archs)} sweep archs: {matched}")

    # ══════════════════════════════════════════════════════════════
    # RE-ESTIMATE ε* WITH SIGMOID FIT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SIGMOID-FIT ε* ESTIMATION")
    print("="*70)

    sigmoid_results = {}
    for arch_name, d in sorted(dense_data.items()):
        eps_vals = d['epsilon_values']
        fg_means = d['forgetting_means']

        eps_star, k, f_min, f_max, r_sq = sigmoid_fit_eps_star(eps_vals, fg_means)

        # Bootstrap
        if 'forgetting_per_seed' in d:
            boot_mean, boot_std, ci_lo, ci_hi = bootstrap_eps_star_sigmoid(eps_vals, d['forgetting_per_seed'])
        else:
            boot_mean, boot_std, ci_lo, ci_hi = eps_star, 0.0, eps_star, eps_star

        sigmoid_results[arch_name] = {
            'eps_star_sigmoid': eps_star,
            'sigmoid_k': k,
            'sigmoid_f_min': f_min,
            'sigmoid_f_max': f_max,
            'sigmoid_r_sq': r_sq,
            'boot_mean': boot_mean,
            'boot_std': boot_std,
            'boot_ci95': [ci_lo, ci_hi],
            'n_params': d.get('n_params', 0),
        }

        print(f"  {arch_name:<20s}: ε* = {eps_star:.3f} (k={k:.2f}, R²={r_sq:.4f})")
        print(f"    F_range: [{f_min:.4f}, {f_max:.4f}]")
        print(f"    Bootstrap: {boot_mean:.3f} ± {boot_std:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save
    with open(os.path.join(RESULTS_DIR, 'sigmoid_eps_star.json'), 'w') as f:
        json.dump(sigmoid_results, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    # NORMALIZATION ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("NORMALIZATION COLLAPSE ANALYSIS (sigmoid ε*)")
    print("="*70)

    common = [a for a in sigmoid_results if a in curv_data]
    if not common:
        print("No overlap between sigmoid results and curvature data")
        return

    eps_stars_raw = {a: sigmoid_results[a]['eps_star_sigmoid'] for a in common}
    raw_vals = list(eps_stars_raw.values())
    raw_mean = np.mean(raw_vals)
    raw_std = np.std(raw_vals)
    raw_cv = raw_std / max(raw_mean, 1e-10)

    print(f"\n  Raw ε* statistics:")
    print(f"    Mean: {raw_mean:.4f} ± {raw_std:.4f}")
    print(f"    CV: {raw_cv:.4f}")
    print(f"    Range: [{min(raw_vals):.4f}, {max(raw_vals):.4f}]")

    # Extract curvature metrics
    metrics = {}
    for a in common:
        c = curv_data[a]
        ht = c['hessian_trace']['mean'] if isinstance(c['hessian_trace'], dict) else c['hessian_trace']
        ft = c['fisher_trace']['mean'] if isinstance(c['fisher_trace'], dict) else c['fisher_trace']
        sn = c['spectral_norm']['mean'] if isinstance(c['spectral_norm'], dict) else c['spectral_norm']
        de = c['d_eff']['mean'] if isinstance(c['d_eff'], dict) else c['d_eff']
        gn = c['gradient_norm']['mean'] if isinstance(c['gradient_norm'], dict) else c['gradient_norm']
        np_ = c['n_params']
        metrics[a] = {
            'hessian_trace': ht, 'fisher_trace': ft, 'spectral_norm': sn,
            'd_eff': de, 'gradient_norm': gn, 'n_params': np_,
            'log_hessian': math.log(ht), 'log_params': math.log(np_),
            'kappa': ht / np_, 'kappa_f': ft / np_,
        }

    # Test normalizations
    normalizations = {
        'raw': lambda a: eps_stars_raw[a],
        'eps * tr(F)': lambda a: eps_stars_raw[a] * metrics[a]['fisher_trace'],
        'eps * tr(H)': lambda a: eps_stars_raw[a] * metrics[a]['hessian_trace'],
        'eps * d_eff': lambda a: eps_stars_raw[a] * metrics[a]['d_eff'],
        'eps * kappa': lambda a: eps_stars_raw[a] * metrics[a]['kappa'],
        'eps * spectral': lambda a: eps_stars_raw[a] * metrics[a]['spectral_norm'],
        'eps * grad^2': lambda a: eps_stars_raw[a] * metrics[a]['gradient_norm']**2,
        'eps * sqrt(tr(H))': lambda a: eps_stars_raw[a] * math.sqrt(metrics[a]['hessian_trace']),
        'eps * sqrt(tr(F))': lambda a: eps_stars_raw[a] * math.sqrt(metrics[a]['fisher_trace']),
        'eps / log(params)': lambda a: eps_stars_raw[a] / math.log(metrics[a]['n_params']),
        'eps * tr(F)/d': lambda a: eps_stars_raw[a] * metrics[a]['kappa_f'],
    }

    print(f"\n  {'Normalization':<25s} {'Mean':>10s} {'Std':>10s} {'CV':>10s} {'vs raw':>10s}")
    print(f"  {'-'*65}")

    norm_results = {}
    for name, fn in normalizations.items():
        vals = [fn(a) for a in common]
        cv = np.std(vals) / max(np.mean(vals), 1e-10)
        relative = cv / max(raw_cv, 1e-10)
        direction = "BETTER" if cv < raw_cv else "WORSE"
        norm_results[name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'cv': float(cv),
            'relative_cv': float(relative),
            'values': {a: float(v) for a, v in zip(common, vals)},
        }
        marker = "✓" if cv < raw_cv else "✗"
        print(f"  {name:<25s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{cv:>10.4f} {relative:>8.2f}× {marker} {direction}")

    # ══════════════════════════════════════════════════════════════
    # CORRELATION ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  ── Correlation: ε* vs Curvature Metrics ──")
    y = np.array([eps_stars_raw[a] for a in common])

    for mname in ['hessian_trace', 'fisher_trace', 'spectral_norm', 'd_eff', 'n_params', 'gradient_norm']:
        x = np.array([metrics[a][mname] for a in common])
        if sp_stats:
            r, p = sp_stats.pearsonr(x, y)
            tau, p_tau = sp_stats.kendalltau(x, y)
        else:
            r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else 0
            p, tau, p_tau = 1.0, 0.0, 1.0
        sig = "**" if p < 0.05 else "  "
        print(f"    ε* vs {mname:<20s}: r={r:+.4f} (p={p:.4f}{sig}), τ={tau:+.4f}")

    # ══════════════════════════════════════════════════════════════
    # POWER LAW FITS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  ── Power Law: ε* = c · metric^α ──")
    for mname in ['hessian_trace', 'fisher_trace', 'spectral_norm', 'd_eff', 'n_params']:
        x = np.log([metrics[a][mname] for a in common])
        y_log = np.log([eps_stars_raw[a] for a in common])
        if sp_stats and np.std(y_log) > 1e-10:
            slope, intercept, r_val, p_val, se = sp_stats.linregress(x, y_log)
        else:
            slope, r_val, p_val, se = 0.0, 0.0, 1.0, 0.0
        print(f"    ε* ∝ {mname:<20s}^α: α={slope:+.4f}±{se:.4f}, R²={r_val**2:.4f}")

    # ══════════════════════════════════════════════════════════════
    # CONSTANCY TEST
    # ══════════════════════════════════════════════════════════════
    print(f"\n  ── Constancy Test ──")
    boot_stds = {a: sigmoid_results[a]['boot_std'] for a in common}
    between_var = np.var(raw_vals)
    within_var = np.mean([s**2 for s in boot_stds.values()])
    f_ratio = between_var / max(within_var, 1e-15)

    if sp_stats:
        df1 = len(raw_vals) - 1
        df2 = 1000
        p_const = 1.0 - sp_stats.f.cdf(f_ratio, df1, df2)
    else:
        p_const = 1.0

    print(f"    Between-arch var: {between_var:.6f}")
    print(f"    Within-arch var:  {within_var:.6f}")
    print(f"    F-ratio: {f_ratio:.4f}")
    print(f"    p-value: {p_const:.6f}")
    if p_const > 0.05:
        print(f"    → CANNOT REJECT H₀: ε* is constant across architectures")
    else:
        print(f"    → REJECT H₀: ε* varies significantly across architectures")

    # Save all analysis
    analysis = {
        'sigmoid_eps_stars': {a: sigmoid_results[a]['eps_star_sigmoid'] for a in common},
        'raw_cv': float(raw_cv),
        'normalizations': norm_results,
        'constancy_test': {
            'between_var': float(between_var),
            'within_var': float(within_var),
            'f_ratio': float(f_ratio),
            'p_value': float(p_const),
            'is_constant': bool(p_const > 0.05),
        },
    }
    with open(os.path.join(RESULTS_DIR, 'post_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.floating, np.integer)) else o)

    # ══════════════════════════════════════════════════════════════
    # GENERATE IMPROVED PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    generate_plots(dense_data, sigmoid_results, curv_data, norm_results, common)


def generate_plots(dense_data, sigmoid_results, curv_data, norm_results, common):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({
            'font.size': 11, 'figure.dpi': 300, 'font.family': 'serif',
            'axes.linewidth': 0.8
        })
    except ImportError:
        print("  matplotlib not available")
        return

    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(dense_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # ── FIGURE 1: Dense forgetting curves with sigmoid fit ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, (arch_name, d) in enumerate(sorted(dense_data.items(),
                                                  key=lambda x: x[1].get('n_params', 0))):
        eps_star = sigmoid_results.get(arch_name, {}).get('eps_star_sigmoid', 0)
        ax.semilogx(d['epsilon_values'], d['forgetting_means'],
                    f'{markers[idx % len(markers)]}-', color=colors[idx],
                    label=f"{arch_name} (ε*={eps_star:.2f})", lw=1.5, ms=5)
        if d.get('forgetting_stds'):
            ax.fill_between(d['epsilon_values'],
                           [m-s for m,s in zip(d['forgetting_means'], d['forgetting_stds'])],
                           [m+s for m,s in zip(d['forgetting_means'], d['forgetting_stds'])],
                           alpha=0.15, color=colors[idx])

        # Overlay sigmoid fit
        if arch_name in sigmoid_results:
            sr = sigmoid_results[arch_name]
            if sr['sigmoid_r_sq'] > 0.5:
                from scipy.optimize import curve_fit
                eps_fine = np.logspace(np.log10(min(d['epsilon_values'])),
                                       np.log10(max(d['epsilon_values'])), 200)
                log_fine = np.log(eps_fine)
                fg_fit = sr['sigmoid_f_min'] + (sr['sigmoid_f_max'] - sr['sigmoid_f_min']) / (
                    1 + np.exp(-sr['sigmoid_k'] * (log_fine - np.log(sr['eps_star_sigmoid']))))
                ax.semilogx(eps_fine, fg_fit, '--', color=colors[idx], alpha=0.5, lw=1)

    ax.set_xlabel('Stability Budget ε (log scale)')
    ax.set_ylabel('Forgetting F')
    ax.set_title('Dense ε Sweep with Sigmoid Fits (FTR, 5 seeds)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sigmoid_forgetting_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'sigmoid_forgetting_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ sigmoid_forgetting_curves")

    # ── FIGURE 2: ε* with bootstrap CI ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    arch_sorted = sorted(sigmoid_results.keys(),
                         key=lambda x: sigmoid_results[x].get('n_params', 0))
    x_pos = range(len(arch_sorted))
    stars = [sigmoid_results[a]['eps_star_sigmoid'] for a in arch_sorted]
    ci_lo = [sigmoid_results[a]['boot_ci95'][0] for a in arch_sorted]
    ci_hi = [sigmoid_results[a]['boot_ci95'][1] for a in arch_sorted]
    errs = [[s-l for s,l in zip(stars, ci_lo)], [h-s for s,h in zip(stars, ci_hi)]]

    ax.errorbar(x_pos, stars, yerr=errs, fmt='ko', capsize=5, ms=8, capthick=1.5)
    ax.axhline(np.mean(stars), color='red', ls='--', lw=1.5, alpha=0.7,
               label=f'Mean = {np.mean(stars):.2f}')
    ax.fill_between([-0.5, len(arch_sorted)-0.5],
                    np.mean(stars)-np.std(stars), np.mean(stars)+np.std(stars),
                    alpha=0.1, color='red')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(arch_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ε* (sigmoid fit)')
    ax.set_title('Critical ε* with 95% Bootstrap CI')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'eps_star_sigmoid_ci.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'eps_star_sigmoid_ci.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ eps_star_sigmoid_ci")

    # ── FIGURE 3: 2D Phase Diagram ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    regime_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    regime_labels = {0: 'Stable (F<0.12)', 1: 'Partial (0.12≤F<0.20)', 2: 'Catastrophic (F≥0.20)'}

    for ax_idx, (metric_name, xlabel) in enumerate([
        ('hessian_trace', 'Hessian Trace tr(H)'),
        ('spectral_norm', 'Spectral Norm ||H||'),
        ('n_params', 'Parameters d')
    ]):
        ax = axes[ax_idx]
        for arch_name in common:
            if arch_name not in dense_data:
                continue
            d = dense_data[arch_name]
            c = curv_data[arch_name]
            m_val = c[metric_name] if not isinstance(c[metric_name], dict) else c[metric_name]['mean']

            for i, eps in enumerate(d['epsilon_values']):
                fg = d['forgetting_means'][i]
                regime = 0 if fg < 0.12 else (1 if fg < 0.20 else 2)
                ax.scatter(m_val, eps, c=regime_colors[regime],
                          s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('ε')
        ax.set_yscale('log')
        if metric_name != 'n_params':
            ax.set_xscale('log')
        ax.set_title(f'Phase Diagram: {xlabel}')
        ax.grid(True, alpha=0.3)
        for r in [0, 1, 2]:
            ax.scatter([], [], c=regime_colors[r], label=regime_labels[r], s=40, edgecolors='black')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'phase_diagram_2d_sigmoid.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'phase_diagram_2d_sigmoid.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ phase_diagram_2d")

    # ── FIGURE 4: Normalization variance bar chart ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    raw_cv = norm_results['raw']['cv']
    norm_names = sorted([k for k in norm_results if k != 'raw'],
                        key=lambda k: norm_results[k]['cv'])
    cvs = [norm_results[n]['cv'] for n in norm_names]
    bar_colors = ['#2ecc71' if c < raw_cv else '#e74c3c' for c in cvs]
    ax.barh(range(len(norm_names)), cvs, color=bar_colors, edgecolor='black', alpha=0.8)
    ax.axvline(x=raw_cv, color='blue', linestyle='--', lw=2, label=f'Raw CV = {raw_cv:.4f}')
    ax.set_yticks(range(len(norm_names)))
    ax.set_yticklabels(norm_names, fontsize=8)
    ax.set_xlabel('Coefficient of Variation')
    ax.set_title('Normalization Collapse Test: Does Rescaling ε* Reduce Spread?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'normalization_collapse.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'normalization_collapse.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ normalization_collapse")

    # ── FIGURE 5: ε* vs curvature scatter ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, (mname, xlabel) in enumerate([
        ('hessian_trace', 'Hessian Trace'),
        ('spectral_norm', 'Spectral Norm'),
        ('n_params', 'Parameters')
    ]):
        ax = axes[ax_idx]
        for a in common:
            c = curv_data[a]
            m_val = c[mname] if not isinstance(c[mname], dict) else c[mname]['mean']
            es = sigmoid_results[a]['eps_star_sigmoid']
            ci = sigmoid_results[a]['boot_ci95']
            ax.errorbar(m_val, es, yerr=[[es-ci[0]], [ci[1]-es]],
                       fmt='o', ms=8, capsize=4, color='steelblue', ecolor='gray')
            ax.annotate(a.replace('CNN_', '').replace('ResNet18_', 'RN'),
                       (m_val, es), fontsize=7, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('ε*')
        ax.set_title(f'ε* vs {xlabel}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'eps_star_vs_curvature.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'eps_star_vs_curvature.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ eps_star_vs_curvature")

    # ── FIGURE 6: Summary (6-panel) ──
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # A: Forgetting curves
    ax = fig.add_subplot(gs[0, 0])
    for idx, (aname, d) in enumerate(sorted(dense_data.items(),
                                             key=lambda x: x[1].get('n_params', 0))):
        ax.semilogx(d['epsilon_values'], d['forgetting_means'],
                    f'{markers[idx%8]}-', color=colors[idx], label=aname, lw=1.2, ms=3)
    ax.set_xlabel('ε'); ax.set_ylabel('Forgetting')
    ax.set_title('(A) FTR Forgetting'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # B: ε* with CI
    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(range(len(arch_sorted)), stars, yerr=errs, fmt='ko', capsize=4, ms=6)
    ax.axhline(np.mean(stars), color='red', ls='--', lw=1.5)
    ax.set_xticks(range(len(arch_sorted)))
    ax.set_xticklabels(arch_sorted, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('ε*'); ax.set_title('(B) ε* ± 95% CI'); ax.grid(True, alpha=0.3, axis='y')

    # C: Phase diagram
    ax = fig.add_subplot(gs[0, 2])
    for aname in common:
        if aname not in dense_data: continue
        d = dense_data[aname]
        c = curv_data[aname]
        ht = c['hessian_trace']['mean'] if isinstance(c['hessian_trace'], dict) else c['hessian_trace']
        for i, eps in enumerate(d['epsilon_values']):
            fg = d['forgetting_means'][i]
            regime = 0 if fg < 0.12 else (1 if fg < 0.20 else 2)
            ax.scatter(ht, eps, c=regime_colors[regime], s=15, alpha=0.7, edgecolors='black', lw=0.2)
    ax.set_xlabel('tr(H)'); ax.set_ylabel('ε'); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('(C) Phase Diagram'); ax.grid(True, alpha=0.3)

    # D: Normalization
    ax = fig.add_subplot(gs[1, 0])
    top_n = sorted([k for k in norm_results if k != 'raw'],
                   key=lambda k: norm_results[k]['cv'])[:6]
    cvs_top = [norm_results[n]['cv'] for n in top_n]
    bc = ['#2ecc71' if c < raw_cv else '#e74c3c' for c in cvs_top]
    ax.barh(range(len(top_n)), cvs_top, color=bc, edgecolor='black')
    ax.axvline(raw_cv, color='blue', ls='--', lw=2)
    ax.set_yticks(range(len(top_n))); ax.set_yticklabels(top_n, fontsize=7)
    ax.set_xlabel('CV'); ax.set_title('(D) Normalization Test'); ax.grid(True, alpha=0.3, axis='x')

    # E: ε* vs Hessian
    ax = fig.add_subplot(gs[1, 1])
    for a in common:
        c = curv_data[a]
        ht = c['hessian_trace']['mean'] if isinstance(c['hessian_trace'], dict) else c['hessian_trace']
        ax.scatter(ht, sigmoid_results[a]['eps_star_sigmoid'], s=60, edgecolors='black', zorder=3)
        ax.annotate(a.replace('CNN_', '').replace('ResNet18_', 'RN'),
                   (ht, sigmoid_results[a]['eps_star_sigmoid']), fontsize=6, ha='center', va='bottom')
    ax.set_xlabel('tr(H)'); ax.set_ylabel('ε*')
    ax.set_title('(E) ε* vs Hessian Trace'); ax.grid(True, alpha=0.3)

    # F: Forgetting surface heatmap
    ax = fig.add_subplot(gs[1, 2])
    for a in common:
        if a not in dense_data: continue
        c = curv_data[a]
        ht = c['hessian_trace']['mean'] if isinstance(c['hessian_trace'], dict) else c['hessian_trace']
        d = dense_data[a]
        sc = ax.scatter([ht]*len(d['epsilon_values']), d['epsilon_values'],
                       c=d['forgetting_means'], cmap='RdYlGn_r', vmin=0, vmax=0.3,
                       s=15, edgecolors='black', lw=0.2)
    ax.set_xlabel('tr(H)'); ax.set_ylabel('ε'); ax.set_xscale('log'); ax.set_yscale('log')
    plt.colorbar(sc, ax=ax, label='Forgetting')
    ax.set_title('(F) Forgetting Heatmap'); ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plots_dir, 'summary_figure.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'summary_figure.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ summary_figure")

    print(f"\n  All plots saved to {plots_dir}")


if __name__ == '__main__':
    main()
