# ============================================================================
# Publication-Quality Visualization for FTR Research
#
# Generates all figures required for a NeurIPS/TPAMI submission:
#   1. Comparative bar charts (accuracy, forgetting, BWT across methods)
#   2. Learning curves with confidence intervals
#   3. Forgetting curves per task
#   4. Lambda dynamics over training
#   5. Drift trajectories
#   6. CKA similarity evolution
#   7. Accuracy-vs-drift Pareto frontiers
#   8. Stability-plasticity tradeoff curves
#   9. Task accuracy heatmaps
#   10. Ablation result grids
#
# All plots use:
#   - Serif fonts (Computer Modern / Times)
#   - 300 DPI
#   - Wong 2011 colorblind-safe palette
#   - Properly labeled axes with units
#   - 95% CI shading
#   - High-res PNG + PDF export
# ============================================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Global Style Configuration
# ============================================================================

# Wong 2011 colorblind-safe palette
COLORS = {
    'baseline':         '#000000',  # black
    'weight_decay':     '#E69F00',  # orange
    'ewc':              '#56B4E9',  # sky blue
    'si':               '#009E73',  # bluish green
    'lwf':              '#F0E442',  # yellow
    'distillation':     '#0072B2',  # blue
    'replay':           '#D55E00',  # vermillion
    'functional_trust': '#CC79A7',  # reddish purple
    'feature_trust':    '#999999',  # gray
    'kl_trust':         '#882255',  # wine
    'ftr_replay':       '#AA4499',  # purple
}

METHOD_LABELS = {
    'baseline':         'Vanilla Fine-tuning',
    'weight_decay':     'Weight Decay',
    'ewc':              'EWC (Kirkpatrick+17)',
    'si':               'SI (Zenke+17)',
    'lwf':              'LwF (Li & Hoiem 16)',
    'distillation':     'Fixed Distillation',
    'replay':           'Replay (buffer=500)',
    'functional_trust': 'FTR (Ours)',
    'feature_trust':    'Feature-Space Trust',
    'kl_trust':         'KL Trust Region',
    'ftr_replay':       'FTR + Replay (Ours)',
}

METHOD_MARKERS = {
    'baseline': 'o', 'weight_decay': 's', 'ewc': '^', 'si': 'v',
    'lwf': 'D', 'distillation': 'P', 'replay': 'X',
    'functional_trust': '*', 'feature_trust': 'h', 'kl_trust': 'p',
    'ftr_replay': 'H',
}

# Display order (our methods last for emphasis)
METHOD_ORDER = [
    'baseline', 'weight_decay', 'ewc', 'si', 'lwf',
    'distillation', 'replay', 'feature_trust', 'functional_trust', 'ftr_replay',
]


def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.figsize': (7, 4.5),
        'axes.prop_cycle': plt.cycler(color=list(COLORS.values())),
    })


def save_figure(fig, path: str, formats: List[str] = None):
    """Save figure in multiple formats."""
    if formats is None:
        formats = ['png', 'pdf']
    base = os.path.splitext(path)[0]
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
    plt.close(fig)


# ============================================================================
# Figure 1: Main Results — Comparative Bar Charts
# ============================================================================

def plot_main_results_bars(
    aggregated: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Bar chart comparing all methods on 4 metrics:
    Average Accuracy (↑), Forgetting (↓), BWT (↑), FWT (↑)
    """
    setup_style()
    
    data = aggregated.get(benchmark, {})
    methods = [m for m in METHOD_ORDER if m in data]
    if not methods:
        return

    metrics = ['average_accuracy', 'forgetting', 'backward_transfer', 'forward_transfer']
    metric_labels = ['Average Accuracy (↑)', 'Forgetting (↓)', 
                     'Backward Transfer (↑)', 'Forward Transfer (↑)']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        means = []
        stds = []
        colors = []
        labels = []
        
        for method in methods:
            mdata = data[method].get(metric, {})
            means.append(mdata.get('mean', 0))
            stds.append(mdata.get('std', 0))
            colors.append(COLORS.get(method, '#333333'))
            labels.append(METHOD_LABELS.get(method, method))

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        
        # Highlight our methods
        for our_method in ['functional_trust', 'ftr_replay']:
            if our_method in methods:
                idx = methods.index(our_method)
                bars[idx].set_edgecolor(COLORS.get(our_method, '#CC79A7'))
                bars[idx].set_linewidth(2)
                bars[idx].set_alpha(1.0)

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'{benchmark.replace("_", " ").title()} — Method Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_main_results.png'))


# ============================================================================
# Figure 2: Forgetting Curves
# ============================================================================

def plot_forgetting_curves(
    all_results: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Line plot showing accuracy on each task over time (across task training).
    One subplot per method, showing how old task accuracy degrades.
    """
    setup_style()
    
    data = all_results.get(benchmark, {})
    methods = [m for m in METHOD_ORDER if m in data]
    if not methods:
        return

    n_methods = len(methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        
        method_data = data[method]
        if 'accuracy_matrix_mean' in method_data:
            acc_mat = np.array(method_data['accuracy_matrix_mean'])
            n_tasks = acc_mat.shape[0]
            
            for task_j in range(n_tasks):
                # Task j becomes visible after being trained (from task j onwards)
                task_accs = acc_mat[task_j:, task_j]
                x_vals = list(range(task_j, n_tasks))
                ax.plot(x_vals, task_accs, marker='o', markersize=4,
                        label=f'Task {task_j}', linewidth=1.5)

            ax.set_xlabel('After Training on Task')
            ax.set_ylabel('Test Accuracy')
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, ncol=2)
            ax.set_xticks(range(n_tasks))

    # Remove empty subplots
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f'{benchmark.replace("_", " ").title()} — Task Forgetting Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_forgetting_curves.png'))


# ============================================================================
# Figure 3: Lambda Dynamics (FTR-specific)
# ============================================================================

def plot_lambda_dynamics(
    all_results: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Two-panel figure showing:
    - Top: Lagrange multiplier λ over training steps
    - Bottom: Functional drift D_f over training steps with ε threshold
    
    Shows the self-regulating dynamics of the dual variable.
    """
    setup_style()
    
    data = all_results.get(benchmark, {})
    if 'functional_trust' not in data:
        return

    ftr_results = data['functional_trust']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # If we have per-seed data, plot mean ± CI
    if isinstance(ftr_results, list):
        # Multiple seeds
        for i, result in enumerate(ftr_results):
            _plot_single_lambda(ax1, ax2, result, alpha=0.3, label=f'Seed {result.get("seed", i)}')
    elif isinstance(ftr_results, dict):
        # Check for raw results
        if 'lambda_history' in ftr_results:
            _plot_single_lambda(ax1, ax2, ftr_results, alpha=0.8, label='FTR')

    ax1.set_ylabel(r'$\lambda$ (Lagrange multiplier)', fontsize=12)
    ax1.set_title('Dual Variable Dynamics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel(r'$D_f$ (Functional Drift)', fontsize=12)
    ax2.set_title('Functional Drift Trajectory', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_lambda_dynamics.png'))


def _plot_single_lambda(ax1, ax2, result: Dict, alpha: float = 0.8, label: str = ''):
    """Plot lambda and drift for a single run."""
    lambdas = result.get('lambda_history', [])
    drifts = result.get('drift_history', [])
    steps = result.get('per_step_metrics', [])
    
    if lambdas:
        ax1.plot(lambdas, alpha=alpha, label=label, color=COLORS['functional_trust'], linewidth=1)
    
    if drifts:
        ax2.plot(drifts, alpha=alpha, label=label, color=COLORS['functional_trust'], linewidth=1)
    
    # Add task boundaries
    boundaries = result.get('task_boundaries', [])
    for b in boundaries:
        ax1.axvline(x=b, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
        ax2.axvline(x=b, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

    # Add epsilon line if available
    if steps:
        epsilons = [s.get('epsilon', None) for s in steps if 'epsilon' in s]
        if epsilons:
            ax2.axhline(y=epsilons[0], color='red', linestyle='--', alpha=0.5, label=r'$\epsilon$')


# ============================================================================
# Figure 4: Stability-Plasticity Tradeoff Frontier
# ============================================================================

def plot_stability_plasticity_frontier(
    aggregated: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Scatter plot with accuracy on y-axis and forgetting on x-axis.
    Each method is a point (with error bars).
    The Pareto frontier is highlighted.
    
    Ideal: top-left corner (high accuracy, low forgetting).
    """
    setup_style()
    
    data = aggregated.get(benchmark, {})
    methods = [m for m in METHOD_ORDER if m in data]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    
    points = []
    for method in methods:
        mdata = data[method]
        acc = mdata.get('average_accuracy', {})
        fgt = mdata.get('forgetting', {})
        
        if acc and fgt:
            ax.errorbar(
                fgt.get('mean', 0), acc.get('mean', 0),
                xerr=fgt.get('std', 0), yerr=acc.get('std', 0),
                fmt=METHOD_MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'),
                markersize=10, capsize=3, linewidth=1.5,
                label=METHOD_LABELS.get(method, method),
                markeredgewidth=1.5 if method == 'functional_trust' else 1,
                markeredgecolor='black' if method == 'functional_trust' else COLORS.get(method, '#333'),
                zorder=10 if method == 'functional_trust' else 5,
            )
            points.append((fgt.get('mean', 0), acc.get('mean', 0), method))

    # Draw Pareto frontier
    if points:
        points_sorted = sorted(points, key=lambda p: p[0])
        pareto_x = [points_sorted[0][0]]
        pareto_y = [points_sorted[0][1]]
        best_y = points_sorted[0][1]
        
        for x, y, _ in points_sorted[1:]:
            if y >= best_y:
                pareto_x.append(x)
                pareto_y.append(y)
                best_y = y
        
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.5, 
                    linewidth=1, label='Pareto Frontier')

    ax.set_xlabel('Forgetting (↓ better)', fontsize=12)
    ax.set_ylabel('Average Accuracy (↑ better)', fontsize=12)
    ax.set_title(f'{benchmark.replace("_", " ").title()} — Stability-Plasticity Tradeoff',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    # Add arrow pointing to ideal corner
    ax.annotate('Ideal', xy=(0, 1), fontsize=9, color='green', alpha=0.5,
                xytext=(0.05, 0.95), textcoords='axes fraction')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_pareto_frontier.png'))


# ============================================================================
# Figure 5: Task Accuracy Heatmap
# ============================================================================

def plot_accuracy_heatmap(
    aggregated: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Heatmap of accuracy matrix for each method.
    Rows = "after training on task i", Columns = "accuracy on task j"
    """
    setup_style()
    
    data = aggregated.get(benchmark, {})
    methods_with_matrix = [m for m in METHOD_ORDER if m in data and 'accuracy_matrix_mean' in data[m]]
    if not methods_with_matrix:
        return

    n = len(methods_with_matrix)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    
    for idx, method in enumerate(methods_with_matrix):
        ax = axes[0][idx]
        mat = np.array(data[method]['accuracy_matrix_mean'])
        
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='equal')
        n_tasks = mat.shape[0]
        
        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_tasks):
                if j <= i:  # Only annotate valid cells
                    val = mat[i, j]
                    color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=color)

        ax.set_xlabel('Task', fontsize=10)
        ax.set_ylabel('After Training on Task', fontsize=10)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight='bold')
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))

    fig.colorbar(im, ax=axes[0][-1], shrink=0.8, label='Accuracy')
    fig.suptitle(f'{benchmark.replace("_", " ").title()} — Accuracy Matrices',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_accuracy_heatmap.png'))


# ============================================================================
# Figure 6: CKA Similarity Comparison
# ============================================================================

def plot_cka_comparison(
    all_results: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    CKA similarity over training for all methods.
    Shows how well each method preserves internal representations.
    """
    setup_style()
    
    data = all_results.get(benchmark, {})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method in METHOD_ORDER:
        if method not in data:
            continue
        
        results = data[method]
        if isinstance(results, list) and results:
            # Multiple seeds — aggregate
            all_cka = [r.get('cka_history', []) for r in results if r.get('cka_history')]
            if all_cka:
                min_len = min(len(c) for c in all_cka)
                if min_len > 0:
                    stacked = np.array([c[:min_len] for c in all_cka])
                    mean_cka = stacked.mean(axis=0)
                    std_cka = stacked.std(axis=0)
                    x = np.arange(min_len)
                    ax.plot(x, mean_cka, color=COLORS.get(method, '#333'),
                            label=METHOD_LABELS.get(method, method), linewidth=1.5)
                    ax.fill_between(x, mean_cka - std_cka, mean_cka + std_cka,
                                    color=COLORS.get(method, '#333'), alpha=0.15)

    ax.set_xlabel('Measurement Point', fontsize=12)
    ax.set_ylabel('CKA Similarity', fontsize=12)
    ax.set_title(f'{benchmark.replace("_", " ").title()} — Representation Stability (CKA)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_cka_comparison.png'))


# ============================================================================
# Figure 7: Learning Curves with CI
# ============================================================================

def plot_learning_curves(
    all_results: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Training accuracy over steps for all methods, with 95% CI shading.
    """
    setup_style()
    
    data = all_results.get(benchmark, {})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method in METHOD_ORDER:
        if method not in data:
            continue
        
        results = data[method]
        if isinstance(results, list) and results:
            all_accs = []
            for r in results:
                steps_data = r.get('per_step_metrics', [])
                accs = [s.get('accuracy', 0) for s in steps_data]
                if accs:
                    all_accs.append(accs)
            
            if all_accs:
                min_len = min(len(a) for a in all_accs)
                if min_len > 0:
                    stacked = np.array([a[:min_len] for a in all_accs])
                    mean_acc = stacked.mean(axis=0)
                    std_acc = stacked.std(axis=0)
                    ci95 = 1.96 * std_acc / np.sqrt(len(all_accs))
                    
                    # Smooth for readability
                    window = max(1, min_len // 100)
                    if window > 1:
                        mean_acc = np.convolve(mean_acc, np.ones(window)/window, mode='valid')
                        ci95 = np.convolve(ci95, np.ones(window)/window, mode='valid')
                    
                    x = np.arange(len(mean_acc))
                    ax.plot(x, mean_acc, color=COLORS.get(method, '#333'),
                            label=METHOD_LABELS.get(method, method), linewidth=1.5)
                    ax.fill_between(x, mean_acc - ci95, mean_acc + ci95,
                                    color=COLORS.get(method, '#333'), alpha=0.15)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{benchmark.replace("_", " ").title()} — Learning Curves',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_learning_curves.png'))


# ============================================================================
# Figure 8: Ablation Results Grid
# ============================================================================

def plot_ablation_grid(
    ablation_results: Dict,
    ablation_name: str,
    save_dir: str,
):
    """
    Side-by-side bar charts for ablation study:
    Left: Average Accuracy, Right: Forgetting
    """
    setup_style()
    
    if not ablation_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    variants = list(ablation_results.keys())
    accs = []
    fgts = []
    
    for v in variants:
        vdata = ablation_results[v]
        accs.append(vdata.get('average_accuracy', {}).get('mean', 0))
        fgts.append(vdata.get('forgetting', {}).get('mean', 0))

    x = np.arange(len(variants))
    
    bars1 = ax1.bar(x, accs, color=COLORS['functional_trust'], alpha=0.7,
                    edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Average Accuracy (↑)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Accuracy', fontsize=11, fontweight='bold')
    
    bars2 = ax2.bar(x, fgts, color=COLORS['ewc'], alpha=0.7,
                    edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Forgetting (↓)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Forgetting', fontsize=11, fontweight='bold')

    fig.suptitle(f'Ablation: {ablation_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'ablation_{ablation_name}.png'))


# ============================================================================
# Figure 9: Drift Comparison Across Methods
# ============================================================================

def plot_drift_comparison(
    all_results: Dict,
    benchmark: str,
    save_dir: str,
):
    """
    Compare functional drift trajectories across methods.
    """
    setup_style()
    
    data = all_results.get(benchmark, {})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method in METHOD_ORDER:
        if method not in data:
            continue
        
        results = data[method]
        if isinstance(results, list) and results:
            all_drifts = [r.get('drift_history', []) for r in results if r.get('drift_history')]
            if all_drifts:
                min_len = min(len(d) for d in all_drifts)
                if min_len > 0:
                    stacked = np.array([d[:min_len] for d in all_drifts])
                    mean_drift = stacked.mean(axis=0)
                    std_drift = stacked.std(axis=0)
                    x = np.arange(min_len)
                    ax.plot(x, mean_drift, color=COLORS.get(method, '#333'),
                            label=METHOD_LABELS.get(method, method), linewidth=1.5)
                    ax.fill_between(x, 
                                    np.maximum(0, mean_drift - std_drift),
                                    mean_drift + std_drift,
                                    color=COLORS.get(method, '#333'), alpha=0.15)

    ax.set_xlabel('Measurement Point', fontsize=12)
    ax.set_ylabel(r'Functional Drift $D_f$', fontsize=12)
    ax.set_title(f'{benchmark.replace("_", " ").title()} — Functional Drift Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, f'{benchmark}_drift_comparison.png'))


# ============================================================================
# Figure 10: Summary Table as Figure
# ============================================================================

def plot_results_table(
    aggregated: Dict,
    save_dir: str,
):
    """
    Render the main results table as a figure for easy inclusion.
    """
    setup_style()
    
    benchmarks = list(aggregated.keys())
    if not benchmarks:
        return

    # Collect all methods across benchmarks
    all_methods = set()
    for b in benchmarks:
        all_methods.update(aggregated[b].keys())
    methods = [m for m in METHOD_ORDER if m in all_methods]

    # Build table data
    header = ['Method'] + [b.replace('_', '\n') for b in benchmarks]
    cell_text = []
    cell_colors = []
    
    for method in methods:
        row = [METHOD_LABELS.get(method, method)]
        for benchmark in benchmarks:
            mdata = aggregated.get(benchmark, {}).get(method, {})
            acc = mdata.get('average_accuracy', {})
            if acc:
                row.append(f"{acc.get('mean', 0):.3f}±{acc.get('std', 0):.3f}")
            else:
                row.append('—')
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(methods) + 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=cell_text,
        colLabels=header,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(header)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight FTR row
    if 'functional_trust' in methods:
        ftr_idx = methods.index('functional_trust') + 1
        for j in range(len(header)):
            table[ftr_idx, j].set_facecolor('#E2EFDA')

    fig.suptitle('Average Accuracy Across Benchmarks', fontsize=13, fontweight='bold', y=0.95)
    
    os.makedirs(save_dir, exist_ok=True)
    save_figure(fig, os.path.join(save_dir, 'results_summary_table.png'))


# ============================================================================
# Master Figure Generator
# ============================================================================

def generate_all_publication_figures(
    aggregated_results: Dict,
    raw_results: Dict,
    save_dir: str,
):
    """
    Generate all publication figures from experiment results.
    
    Args:
        aggregated_results: Dict[benchmark][method] -> aggregated stats
        raw_results: Dict[benchmark][method] -> list of per-seed results
        save_dir: Output directory for figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    benchmarks = list(aggregated_results.keys())
    
    for benchmark in benchmarks:
        print(f"  Generating figures for {benchmark}...")
        
        # Figure 1: Main results bar chart
        try:
            plot_main_results_bars(aggregated_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: main_results_bars failed: {e}")

        # Figure 2: Forgetting curves
        try:
            plot_forgetting_curves(aggregated_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: forgetting_curves failed: {e}")

        # Figure 3: Lambda dynamics
        try:
            plot_lambda_dynamics(raw_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: lambda_dynamics failed: {e}")

        # Figure 4: Stability-plasticity frontier
        try:
            plot_stability_plasticity_frontier(aggregated_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: pareto_frontier failed: {e}")

        # Figure 5: Accuracy heatmap
        try:
            plot_accuracy_heatmap(aggregated_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: accuracy_heatmap failed: {e}")

        # Figure 6: CKA comparison
        try:
            plot_cka_comparison(raw_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: cka_comparison failed: {e}")

        # Figure 7: Learning curves
        try:
            plot_learning_curves(raw_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: learning_curves failed: {e}")

        # Figure 9: Drift comparison
        try:
            plot_drift_comparison(raw_results, benchmark, save_dir)
        except Exception as e:
            print(f"    Warning: drift_comparison failed: {e}")

    # Figure 10: Summary table
    try:
        plot_results_table(aggregated_results, save_dir)
    except Exception as e:
        print(f"    Warning: results_table failed: {e}")
    
    print(f"  All figures saved to {save_dir}")
