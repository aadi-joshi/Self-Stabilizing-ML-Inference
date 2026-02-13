# ============================================================================
# Publication-Quality Visualization Suite
# 300 DPI, PDF/PNG export, colorblind-friendly, seaborn styling
# ============================================================================

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Publication style setup
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'baseline':         '#E69F00',   # Orange
    'weight_decay':     '#56B4E9',   # Sky blue
    'ewc':              '#009E73',   # Green
    'functional_trust': '#CC79A7',   # Reddish purple
    'kl_trust':         '#0072B2',   # Blue
}

METHOD_LABELS = {
    'baseline':         'Standard Adam',
    'weight_decay':     'Weight Decay',
    'ewc':              'EWC',
    'functional_trust': 'Functional Trust Region (Ours)',
    'kl_trust':         'KL Trust Region',
}

LINESTYLES = {
    'baseline':         '-.',
    'weight_decay':     ':',
    'ewc':              '--',
    'functional_trust': '-',
    'kl_trust':         '--',
}


def load_all_metrics(results_dir: str, experiment: str = None) -> Dict[str, List]:
    """Load metrics for all methods and seeds from a results directory."""
    all_metrics = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('_metrics.json'):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            # Filter by experiment if specified
            if experiment and data.get('experiment', '') != experiment:
                # Also accept if no experiment field, or if the file is in the right dir
                pass  # Accept it anyway — the directory structure provides context
            method = data.get('method', fname.split('_')[0])
            if method not in all_metrics:
                all_metrics[method] = []
            all_metrics[method].append(data)
    return all_metrics


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def aggregate_over_seeds(metrics_list: List[Dict], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a metric across seeds: returns (mean, lower_ci, upper_ci)."""
    all_vals = []
    for m in metrics_list:
        vals = [step[key] for step in m.get('steps', []) if key in step]
        all_vals.append(vals)

    # Truncate to shortest length
    min_len = min(len(v) for v in all_vals) if all_vals else 0
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])

    arr = np.array([v[:min_len] for v in all_vals])
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    n = len(all_vals)
    ci = 1.96 * std / np.sqrt(n) if n > 1 else std

    return mean, mean - ci, mean + ci


# ============================================================================
# Figure 1: Accuracy/Loss curves with confidence bands
# ============================================================================
def plot_learning_curves(
    all_metrics: Dict[str, List],
    metric_key: str = 'accuracy',
    title: str = 'Learning Curves',
    ylabel: str = 'Accuracy',
    save_path: str = None,
    smooth: int = 5,
):
    """Plot learning curves with 95% CI bands for all methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, runs in all_metrics.items():
        mean, lower, upper = aggregate_over_seeds(runs, metric_key)
        if len(mean) == 0:
            continue
        if smooth > 1:
            mean = smooth_curve(mean.tolist(), smooth)
            lower = smooth_curve(lower.tolist(), smooth)
            upper = smooth_curve(upper.tolist(), smooth)

        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        ls = LINESTYLES.get(method, '-')

        ax.plot(x, mean, color=color, linestyle=ls, label=label, zorder=5)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 2: Functional Drift Over Training
# ============================================================================
def plot_drift_curves(
    all_metrics: Dict[str, List],
    title: str = 'Functional Drift',
    save_path: str = None,
    smooth: int = 5,
):
    """Plot functional drift with epsilon thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Drift curves
    ax = axes[0]
    for method, runs in all_metrics.items():
        mean, lower, upper = aggregate_over_seeds(runs, 'functional_drift')
        if len(mean) == 0:
            continue
        if smooth > 1:
            mean = smooth_curve(mean.tolist(), smooth)
            lower = smooth_curve(lower.tolist(), smooth)
            upper = smooth_curve(upper.tolist(), smooth)

        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        ls = LINESTYLES.get(method, '-')
        ax.plot(x, mean, color=color, linestyle=ls, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(r'$D_f(\theta_t, \theta_0)$')
    ax.set_title('Functional Drift')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: Lambda evolution (for methods that have it)
    ax = axes[1]
    for method, runs in all_metrics.items():
        mean, lower, upper = aggregate_over_seeds(runs, 'lambda')
        if len(mean) == 0:
            continue
        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method) + r' $\lambda$'
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(r'$\lambda$ (Lagrange Multiplier)')
    ax.set_title('Dual Variable Evolution')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 3: Task Accuracy Matrix (Continual Learning)
# ============================================================================
def plot_task_accuracy_matrix(
    all_metrics: Dict[str, List],
    methods_to_plot: List[str] = None,
    n_tasks: int = 5,
    save_path: str = None,
):
    """Plot task accuracy matrix showing catastrophic forgetting."""
    if methods_to_plot is None:
        methods_to_plot = list(all_metrics.keys())

    n_methods = len(methods_to_plot)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 3.5))
    if n_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods_to_plot):
        ax = axes[idx]
        runs = all_metrics.get(method, [])
        if not runs:
            continue

        # Build accuracy matrix: rows = tasks learned up to, cols = tasks evaluated on
        # Average across seeds
        matrices = []
        for run in runs:
            task_accs = run.get('task_accuracies', {})
            matrix = np.full((n_tasks, n_tasks), np.nan)
            for task_key, accs in task_accs.items():
                task_id = int(task_key.split('_')[1])
                for step_idx, acc in enumerate(accs):
                    if step_idx < n_tasks and task_id < n_tasks:
                        matrix[step_idx, task_id] = acc
            matrices.append(matrix)

        if matrices:
            avg_matrix = np.nanmean(matrices, axis=0)
        else:
            avg_matrix = np.full((n_tasks, n_tasks), np.nan)

        im = ax.imshow(avg_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel('Evaluated Task')
        ax.set_ylabel('After Training Task')
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))

        # Annotate cells
        for i in range(avg_matrix.shape[0]):
            for j in range(avg_matrix.shape[1]):
                if not np.isnan(avg_matrix[i, j]):
                    text = ax.text(j, i, f'{avg_matrix[i, j]:.2f}',
                                   ha='center', va='center', fontsize=8,
                                   color='black' if avg_matrix[i, j] > 0.5 else 'white')

    fig.colorbar(im, ax=axes, shrink=0.8, label='Accuracy')
    plt.suptitle('Task Accuracy Matrix', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 4: Forgetting Bar Chart
# ============================================================================
def plot_forgetting_comparison(
    all_metrics: Dict[str, List],
    save_path: str = None,
):
    """Bar chart of average forgetting per method with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods_ordered = ['baseline', 'weight_decay', 'ewc', 'kl_trust', 'functional_trust']
    methods = [m for m in methods_ordered if m in all_metrics]

    bar_data = []
    for method in methods:
        forgetting_vals = []
        for run in all_metrics[method]:
            fscores = run.get('forgetting_scores', {})
            for _, scores in fscores.items():
                forgetting_vals.extend(scores)
        if forgetting_vals:
            bar_data.append({
                'method': method,
                'mean': np.mean(forgetting_vals),
                'std': np.std(forgetting_vals),
                'ci': 1.96 * np.std(forgetting_vals) / np.sqrt(max(len(forgetting_vals), 1)),
            })

    x = np.arange(len(bar_data))
    colors = [COLORS.get(d['method'], '#333') for d in bar_data]
    labels = [METHOD_LABELS.get(d['method'], d['method']) for d in bar_data]
    means = [d['mean'] for d in bar_data]
    cis = [d['ci'] for d in bar_data]

    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Average Forgetting')
    ax.set_title('Catastrophic Forgetting Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add significance markers
    if len(bar_data) >= 2:
        # Mark the best method
        best_idx = np.argmin(means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 5: CKA Representation Similarity
# ============================================================================
def plot_cka_curves(
    all_metrics: Dict[str, List],
    title: str = 'Representation Stability (CKA)',
    save_path: str = None,
    smooth: int = 5,
):
    """Plot CKA similarity over training."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, runs in all_metrics.items():
        mean, lower, upper = aggregate_over_seeds(runs, 'cka_similarity')
        if len(mean) == 0:
            continue
        if smooth > 1:
            mean = smooth_curve(mean.tolist(), smooth)
            lower = smooth_curve(lower.tolist(), smooth)
            upper = smooth_curve(upper.tolist(), smooth)

        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        ls = LINESTYLES.get(method, '-')
        ax.plot(x, mean, color=color, linestyle=ls, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('CKA Similarity')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 6: RL Learning Curves
# ============================================================================
def plot_rl_reward_curves(
    all_metrics: Dict[str, List],
    save_path: str = None,
    smooth: int = 50,
):
    """Plot RL reward curves with confidence bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: episode rewards
    ax = axes[0]
    for method, runs in all_metrics.items():
        all_rewards = []
        for run in runs:
            rewards = run.get('episode_rewards', [])
            all_rewards.append(rewards)

        if not all_rewards:
            continue

        min_len = min(len(r) for r in all_rewards)
        arr = np.array([r[:min_len] for r in all_rewards])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        ci = 1.96 * std / np.sqrt(len(all_rewards))

        if smooth > 1:
            mean = smooth_curve(mean.tolist(), smooth)
            lower = smooth_curve((mean - ci[:len(mean)]).tolist(), smooth) if len(ci) >= len(mean) else mean
            upper = smooth_curve((mean + ci[:len(mean)]).tolist(), smooth) if len(ci) >= len(mean) else mean
            # Recalculate properly
            ci_s = smooth_curve(ci[:min_len].tolist(), smooth)
            lower = mean - ci_s[:len(mean)]
            upper = mean + ci_s[:len(mean)]

        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        ls = LINESTYLES.get(method, '-')
        ax.plot(x, mean, color=color, linestyle=ls, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('RL Learning Curves')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: drift
    ax = axes[1]
    for method, runs in all_metrics.items():
        mean, lower, upper = aggregate_over_seeds(runs, 'functional_drift')
        if len(mean) == 0:
            continue
        x = np.arange(len(mean))
        color = COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$D_f(\theta_t, \theta_0)$')
    ax.set_title('Policy Drift')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 7: Ablation Grid
# ============================================================================
def plot_ablation_grid(
    ablation_results: Dict[str, Dict],
    save_path: str = None,
):
    """Plot ablation study results as subplots."""
    categories = list(ablation_results.keys())
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (category, data) in enumerate(ablation_results.items()):
        ax = axes[idx]
        configs = list(data.keys())
        means = [data[c].get('mean_acc', 0) for c in configs]
        stds = [data[c].get('std_acc', 0) for c in configs]
        drifts = [data[c].get('mean_drift', 0) for c in configs]

        x = np.arange(len(configs))
        width = 0.35
        bars1 = ax.bar(x - width / 2, means, width, yerr=stds, label='Accuracy',
                       color='#CC79A7', capsize=3, alpha=0.8)

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, drifts, width, label='Drift',
                        color='#56B4E9', alpha=0.8)

        ax.set_xlabel(category.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Drift')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in configs], rotation=30, ha='right', fontsize=8)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)

    plt.suptitle('Ablation Study', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 8: Pareto Frontier (Accuracy vs Stability)
# ============================================================================
def plot_pareto_frontier(
    all_metrics: Dict[str, List],
    save_path: str = None,
):
    """Scatter plot of accuracy vs drift, highlighting the Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, runs in all_metrics.items():
        accs = []
        drifts = []
        for run in runs:
            steps = run.get('steps', [])
            if steps:
                final_acc = np.mean([s.get('accuracy', 0) for s in steps[-20:]])
                final_drift = np.mean([s.get('functional_drift', 0) for s in steps[-20:]])
                accs.append(final_acc)
                drifts.append(final_drift)

        if accs:
            color = COLORS.get(method, '#333333')
            label = METHOD_LABELS.get(method, method)
            ax.scatter(drifts, accs, c=color, label=label, s=80, alpha=0.8,
                       edgecolors='black', linewidth=0.5, zorder=5)
            # Mean point
            ax.scatter([np.mean(drifts)], [np.mean(accs)], c=color,
                       s=200, marker='*', edgecolors='black', linewidth=1, zorder=10)

    ax.set_xlabel(r'Functional Drift $D_f$')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Accuracy–Stability Trade-off')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1])
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================================
# Comprehensive figure generation
# ============================================================================
def generate_all_figures(results_dir: str, output_dir: str, experiment: str):
    """Generate all figures for a given experiment."""
    from utils.common import ensure_dir
    ensure_dir(output_dir)

    all_metrics = load_all_metrics(results_dir, experiment)
    if not all_metrics:
        print(f"  No metrics found in {results_dir} for {experiment}")
        return

    print(f"  Loaded methods: {list(all_metrics.keys())}")
    print(f"  Seeds per method: {[len(v) for v in all_metrics.values()]}")

    # Learning curves
    plot_learning_curves(all_metrics, 'accuracy',
                         f'{experiment}: Accuracy', 'Accuracy',
                         os.path.join(output_dir, f'{experiment}_accuracy.pdf'))

    # Drift curves
    plot_drift_curves(all_metrics, f'{experiment}: Functional Drift',
                      os.path.join(output_dir, f'{experiment}_drift.pdf'))

    # CKA curves
    plot_cka_curves(all_metrics, f'{experiment}: CKA Similarity',
                    os.path.join(output_dir, f'{experiment}_cka.pdf'))

    # Pareto frontier
    plot_pareto_frontier(all_metrics,
                         os.path.join(output_dir, f'{experiment}_pareto.pdf'))

    if experiment == 'continual_cifar':
        plot_task_accuracy_matrix(all_metrics,
                                  save_path=os.path.join(output_dir, f'{experiment}_task_matrix.pdf'))
        plot_forgetting_comparison(all_metrics,
                                   save_path=os.path.join(output_dir, f'{experiment}_forgetting.pdf'))

    if experiment == 'rl_gridworld':
        plot_rl_reward_curves(all_metrics,
                              save_path=os.path.join(output_dir, f'{experiment}_rl_rewards.pdf'))

    print(f"  All figures generated in {output_dir}")
