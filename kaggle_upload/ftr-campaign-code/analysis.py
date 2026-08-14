"""
Statistical analysis for the FTR phase-transition campaign.

Fixes / additions relative to stability_constrained_selfimprovement/analyze_phase_diagram.py:

  1. The original "F-test for constancy" (analyze_phase_diagram.py:299-322,
     duplicated in run_phase_diagram.py:797-822) computes
     p = 1 - F.cdf(f_ratio, df1=n_arch-1, df2=1000) with df2 HARDCODED to
     1000 regardless of the actual number of architectures or bootstrap
     resamples, while the paper's Table 3 reports "Degrees of freedom (7,7)"
     -- i.e. the manuscript's stated methodology does not match the code
     that produced its own number (df2=1000 gives p=0.786; df2=7 gives
     p=0.767). This is a real inconsistency, not just a debatable
     approximation. `hierarchical_partial_pooling` below replaces this
     entirely with a normal-normal partial-pooling model that has a
     principled likelihood and reports a posterior over the population-level
     epsilon* and an intraclass-correlation-style universality index,
     instead of a single ad hoc p-value.
  2. `leave_one_out` recomputes every summary statistic with each
     architecture dropped in turn (NEXT.md Sec 4.2: a single high-leverage
     point should not be silently trusted).
  3. `correlation_power` reports the minimum detectable |r| at n architectures
     and a Bayes factor (JZS-style approximation) for each curvature
     correlation, instead of relying on a bare p>0.05 (NEXT.md Sec 4.1).
  4. `finite_size_scaling` regresses sigmoid sharpness k against CNN width
     to test k ~ W^alpha (NEXT.md Sec 8 -- the finite-size-scaling novelty
     lever), using only the width-sweep family so depth/BN/family are held
     fixed.
"""
import math
import numpy as np


# ======================================================================
# Sigmoid fit + bootstrap (ported from analyze_phase_diagram.py, unchanged
# logic -- this part of the original pipeline is sound)
# ======================================================================
def sigmoid_fit_eps_star(eps_values, forgetting_values):
    from scipy.optimize import curve_fit

    eps = np.array(eps_values, dtype=float)
    fg = np.array(forgetting_values, dtype=float)
    order = np.argsort(eps)
    eps, fg = eps[order], fg[order]
    log_eps = np.log(eps)

    def logistic(x, f_min, f_max, k, x0):
        return f_min + (f_max - f_min) / (1.0 + np.exp(-k * (x - x0)))

    f_min0, f_max0 = np.min(fg), np.max(fg)
    x0_0, k0 = np.median(log_eps), 1.0

    try:
        popt, _ = curve_fit(
            logistic, log_eps, fg, p0=[f_min0, f_max0, k0, x0_0],
            bounds=([0, 0, 0.01, log_eps[0] - 1], [1, 1, 50, log_eps[-1] + 1]),
            maxfev=10000)
        f_min, f_max, k, log_eps_star = popt
        eps_star = float(np.exp(log_eps_star))
        fg_pred = logistic(log_eps, *popt)
        ss_res = np.sum((fg - fg_pred) ** 2)
        ss_tot = np.sum((fg - np.mean(fg)) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)
        return eps_star, float(k), float(f_min), float(f_max), float(r_sq)
    except Exception:
        return _fallback_finite_diff(eps, fg)


def is_bound_saturated(eps_star, eps_values, tol=0.02):
    """
    Detects when sigmoid_fit_eps_star's point estimate has pinned against the
    curve_fit search bound (log_eps[0]-1, log_eps[-1]+1) rather than locating
    a genuine transition inside the tested grid. This happens for
    near-flat/poorly-identified curves, where the optimizer has no gradient
    signal pulling x0 away from the bound. Two architectures in the original
    30-arch dense sweep (ResNetLite_W8_NoBN, ResNet18_W16) both reported
    eps*=135.91=50*e, i.e. exp(log(max(eps_values))+1) to the exact decimal --
    not a coincidence of independent bootstrap resampling, but this bound
    being hit deterministically on the un-resampled fit and reproduced across
    most resamples. Returns True if eps_star is within `tol` (relative) of
    either bound.
    """
    lo_bound = math.exp(math.log(min(eps_values)) - 1)
    hi_bound = math.exp(math.log(max(eps_values)) + 1)
    return (abs(eps_star - lo_bound) / lo_bound < tol) or (abs(eps_star - hi_bound) / hi_bound < tol)


def _fallback_finite_diff(eps_sorted, fg_sorted):
    log_eps = np.log(eps_sorted)
    derivs = []
    for i in range(1, len(log_eps)):
        d_fg = fg_sorted[i] - fg_sorted[i - 1]
        d_le = log_eps[i] - log_eps[i - 1]
        derivs.append(abs(d_fg / d_le) if abs(d_le) > 1e-10 else 0.0)
    interior = derivs[1:-1] if len(derivs) > 3 else derivs
    offset = 1 if len(derivs) > 3 else 0
    max_idx = int(np.argmax(interior)) + offset
    eps_star = float(math.sqrt(eps_sorted[max_idx] * eps_sorted[max_idx + 1]))
    return eps_star, float(max(derivs)), 0.0, 0.0, 0.0


def bootstrap_eps_star_sigmoid(eps_values, forgetting_per_seed, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
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
            if 0.05 < es < 200:
                boot_stars.append(es)
        except Exception:
            pass
    if len(boot_stars) < 10:
        return np.mean(eps_values), np.std(eps_values), 0.0, 100.0
    bs = np.array(boot_stars)
    return float(np.mean(bs)), float(np.std(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ======================================================================
# Hierarchical partial-pooling model (replaces the buggy F-test)
# ======================================================================
def hierarchical_partial_pooling(eps_star_by_arch, boot_std_by_arch, n_mcmc=20000, burn=4000, seed=0):
    """
    Normal-normal hierarchical model:
        eps_star_a ~ Normal(mu, tau)          [population level]
        hat_eps_a  ~ Normal(eps_star_a, s_a)  [measurement, s_a = bootstrap SE]
    Fit by Gibbs sampling (conjugate updates), giving a posterior over the
    population mean mu, the between-architecture SD tau, and each
    architecture's shrunk estimate. Reports:
      - posterior mean/CI for mu (the "universal" eps*)
      - posterior mean/CI for tau (residual between-architecture spread
        AFTER accounting for measurement noise -- tau near 0 is the
        Bayesian analogue of "cannot reject constancy", without needing an
        arbitrarily chosen bootstrap-vs-F degrees of freedom)
      - an intraclass-style universality index: ICC = tau^2/(tau^2+mean(s_a^2))
        near 0 means nearly all apparent between-arch variation is
        measurement noise; near 1 means real architecture-dependence.
    """
    archs = list(eps_star_by_arch.keys())
    y = np.array([eps_star_by_arch[a] for a in archs], dtype=float)
    s = np.array([max(boot_std_by_arch[a], 1e-6) for a in archs], dtype=float)
    n = len(archs)
    rng = np.random.RandomState(seed)

    mu = float(np.mean(y))
    tau2 = float(np.var(y)) + 1e-6
    theta = y.copy()  # per-architecture latent true eps*

    mu_chain, tau_chain, theta_chain = [], [], []
    for it in range(n_mcmc):
        # theta_a | mu, tau2, y_a, s_a  (conjugate normal-normal)
        prec_prior = 1.0 / tau2
        prec_lik = 1.0 / (s ** 2)
        post_var = 1.0 / (prec_prior + prec_lik)
        post_mean = post_var * (prec_prior * mu + prec_lik * y)
        theta = rng.normal(post_mean, np.sqrt(post_var))

        # mu | theta, tau2  (flat prior)
        mu_var = tau2 / n
        mu = rng.normal(np.mean(theta), math.sqrt(max(mu_var, 1e-12)))

        # tau2 | theta, mu  (inverse-gamma with weak prior a0=1, b0=1e-3)
        a0, b0 = 1.0, 1e-3
        a_post = a0 + n / 2.0
        b_post = b0 + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        if it >= burn:
            mu_chain.append(mu)
            tau_chain.append(math.sqrt(max(tau2, 0.0)))
            theta_chain.append(theta.copy())

    mu_chain = np.array(mu_chain)
    tau_chain = np.array(tau_chain)
    theta_chain = np.array(theta_chain)  # (n_samples, n_arch)

    mean_s2 = float(np.mean(s ** 2))
    icc_chain = tau_chain ** 2 / (tau_chain ** 2 + mean_s2)

    return {
        'architectures': archs,
        'raw_eps_star': dict(zip(archs, y.tolist())),
        'bootstrap_se': dict(zip(archs, s.tolist())),
        'mu_posterior_mean': float(np.mean(mu_chain)),
        'mu_posterior_sd': float(np.std(mu_chain)),
        'mu_ci95': [float(np.percentile(mu_chain, 2.5)), float(np.percentile(mu_chain, 97.5))],
        'tau_posterior_mean': float(np.mean(tau_chain)),
        'tau_posterior_sd': float(np.std(tau_chain)),
        'tau_ci95': [float(np.percentile(tau_chain, 2.5)), float(np.percentile(tau_chain, 97.5))],
        'icc_posterior_mean': float(np.mean(icc_chain)),
        'icc_ci95': [float(np.percentile(icc_chain, 2.5)), float(np.percentile(icc_chain, 97.5))],
        'prob_tau_less_than_1': float(np.mean(tau_chain < 1.0)),
        'shrunk_theta_mean': {a: float(np.mean(theta_chain[:, i])) for i, a in enumerate(archs)},
        'n_mcmc_kept': len(mu_chain),
    }


# ======================================================================
# Leave-one-out sensitivity
# ======================================================================
def leave_one_out(eps_star_by_arch, boot_std_by_arch):
    archs = list(eps_star_by_arch.keys())
    results = {}
    for drop in archs:
        keep = [a for a in archs if a != drop]
        vals = np.array([eps_star_by_arch[a] for a in keep])
        results[f'drop_{drop}'] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'cv': float(np.std(vals) / max(np.mean(vals), 1e-10)),
            'n': len(keep),
        }
    full = np.array([eps_star_by_arch[a] for a in archs])
    results['full'] = {
        'mean': float(np.mean(full)),
        'std': float(np.std(full)),
        'cv': float(np.std(full) / max(np.mean(full), 1e-10)),
        'n': len(archs),
    }
    return results


# ======================================================================
# Correlation power analysis + Bayes factor
# ======================================================================
def _min_detectable_r(n, alpha=0.05, power=0.8):
    """Approximate two-sided min detectable |r| via Fisher z (Cohen 1988 formula)."""
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    z_r = (z_alpha + z_beta) / math.sqrt(n - 3)
    return float(math.tanh(z_r))


def _bf_jzs_correlation(r, n):
    """
    JZS Bayes factor (BF10) for a Pearson correlation, Wetzels & Wagenmakers
    (2012) closed-ish form via numerical integration of the marginal
    likelihood under a Jeffreys prior on rho. Returns BF10 (evidence for a
    non-zero correlation vs the null). Integrates in log-space and clips r
    away from +/-1 to avoid the (1-r^2) singularity blowing up for n>~20.
    """
    import warnings
    from scipy import integrate

    r = float(np.clip(r, -0.999, 0.999))
    n = max(int(n), 4)

    def log_integrand(rho):
        rho = np.clip(rho, -0.999999, 0.999999)
        return ((n - 1) / 2.0) * math.log1p(-rho ** 2) + (-(n - 1) + 0.5) * math.log1p(-rho * r)

    # peak-normalize before exponentiating to keep the integrand well-scaled
    grid = np.linspace(-0.999, 0.999, 400)
    log_vals = np.array([log_integrand(g) for g in grid])
    peak = np.max(log_vals)

    def integrand(rho):
        return math.exp(log_integrand(rho) - peak)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, _ = integrate.quad(integrand, -1, 1, limit=200)
    if not math.isfinite(val) or val <= 0:
        return float('nan')
    # true_integral = val * exp(peak); prior marginal (uniform density 1/2 on [-1,1]) = 2.0
    log_true_integral = math.log(val) + peak
    log_bf10 = log_true_integral - math.log(2.0)
    bf10 = math.exp(log_bf10) if log_bf10 < 700 else float('inf')
    return float(bf10) if math.isfinite(bf10) else float('inf')


def correlation_power(eps_star_by_arch, curvature_by_arch, metrics=None):
    from scipy import stats
    archs = [a for a in eps_star_by_arch if a in curvature_by_arch]
    n = len(archs)
    metrics = metrics or ['hessian_trace', 'fisher_trace', 'spectral_norm', 'd_eff', 'n_params', 'gradient_norm']
    y = np.array([eps_star_by_arch[a] for a in archs])
    out = {'n_architectures': n, 'min_detectable_r_at_80pct_power': _min_detectable_r(n)}
    results = {}
    for m in metrics:
        x = np.array([curvature_by_arch[a][m] for a in archs])
        r, p = stats.pearsonr(x, y)
        try:
            bf10 = _bf_jzs_correlation(r, n)
        except Exception:
            bf10 = float('nan')
        results[m] = {'pearson_r': float(r), 'pearson_p': float(p), 'bf10': bf10,
                       'evidence_for_null': bool(bf10 < 1 / 3) if bf10 == bf10 else None}
    out['correlations'] = results
    return out


# ======================================================================
# Finite-size scaling: sigmoid sharpness k vs CNN width
# ======================================================================
def finite_size_scaling(sigmoid_results, width_arch_names, widths):
    """
    sigmoid_results: dict arch -> {'sigmoid_k':..., 'eps_star_sigmoid':..., ...}
    width_arch_names / widths: parallel lists, e.g. ['CNN_W8',...], [8,16,...]
    Fits log(k) = a + alpha*log(W) (power law k ~ W^alpha) and reports
    whether the transition width Delta_eps = ln(9)/k narrows with width
    (a genuine finite-size-scaling signature) or is flat.
    """
    from scipy import stats
    ks, ws, eps_stars = [], [], []
    for name, w in zip(width_arch_names, widths):
        if name in sigmoid_results:
            ks.append(sigmoid_results[name]['sigmoid_k'])
            ws.append(w)
            eps_stars.append(sigmoid_results[name]['eps_star_sigmoid'])
    if len(ks) < 3:
        return {'error': 'insufficient width-family architectures with sigmoid fits'}

    log_w = np.log(ws)
    log_k = np.log(ks)
    slope, intercept, r, p, se = stats.linregress(log_w, log_k)
    delta_eps = [math.log(9) / k for k in ks]

    return {
        'widths': ws, 'sharpness_k': ks, 'eps_star': eps_stars, 'delta_eps': delta_eps,
        'power_law_alpha': float(slope), 'power_law_alpha_se': float(se),
        'power_law_intercept': float(intercept), 'r_squared': float(r ** 2), 'p_value': float(p),
        'interpretation': (
            'k grows with width (alpha>0): transition sharpens as the network widens, '
            'consistent with a finite-size-scaling crossover toward a sharp transition '
            'in the infinite-width limit.' if slope > 0 else
            'k does not grow with width: no evidence of finite-size sharpening.'
        ),
    }
