# ============================================================================
# Statistical Analysis Pipeline
# Aggregation across seeds, Welch's t-test, Cohen's d, significance tables
# ============================================================================

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metrics.experiment_metrics import StatisticalAnalyzer
from utils.common import ensure_dir


def run_statistical_analysis(
    results_dir: str,
    output_dir: str,
    experiment: str = 'continual_cifar',
    reference_method: str = 'functional_trust',
) -> Dict:
    """
    Run full statistical analysis: aggregate, compare, produce tables.
    
    Args:
        results_dir: Directory containing *_metrics.json files
        output_dir: Where to save tables and reports
        experiment: Which experiment to analyze
        reference_method: Our method to compare others against
        
    Returns:
        Dict with all statistical results
    """
    ensure_dir(output_dir)
    analyzer = StatisticalAnalyzer()

    # Load all results grouped by method
    method_results = _load_method_results(results_dir)
    if not method_results:
        print(f"No results found in {results_dir}")
        return {}

    print(f"Loaded methods: {list(method_results.keys())}")
    for m, runs in method_results.items():
        print(f"  {m}: {len(runs)} seeds")

    results = {
        'experiment': experiment,
        'reference_method': reference_method,
        'method_summaries': {},
        'comparisons': {},
        'tables': {},
    }

    # 1. Per-method aggregation
    for method, runs in method_results.items():
        aggregated = _aggregate_method(runs)
        results['method_summaries'][method] = aggregated
        print(f"\n{method}:")
        for key, val in aggregated.items():
            print(f"  {key}: {val.get('mean', 0):.4f} ± {val.get('std', 0):.4f} "
                  f"[{val.get('ci_lower', 0):.4f}, {val.get('ci_upper', 0):.4f}]")

    # 2. Pairwise comparisons against reference
    if reference_method in method_results:
        ref_runs = method_results[reference_method]
        for method, runs in method_results.items():
            if method == reference_method:
                continue

            comparison = _compare_methods(ref_runs, runs, method)
            results['comparisons'][f"{reference_method}_vs_{method}"] = comparison

            print(f"\n{reference_method} vs {method}:")
            for metric, stats in comparison.items():
                sig = "***" if stats.get('p_value', 1) < 0.001 else \
                      "**" if stats.get('p_value', 1) < 0.01 else \
                      "*" if stats.get('p_value', 1) < 0.05 else "ns"
                print(f"  {metric}: t={stats.get('t_stat', 0):.3f}, "
                      f"p={stats.get('p_value', 1):.4f}, "
                      f"d={stats.get('cohens_d', 0):.3f} {sig}")

    # 3. Generate LaTeX table
    latex_table = _generate_latex_table(results)
    results['tables']['main'] = latex_table

    latex_path = os.path.join(output_dir, f'{experiment}_results_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")

    # 4. Generate comparison table
    comparison_table = _generate_comparison_table(results)
    results['tables']['comparison'] = comparison_table

    comp_path = os.path.join(output_dir, f'{experiment}_comparison_table.tex')
    with open(comp_path, 'w') as f:
        f.write(comparison_table)

    # 5. Save full results
    report_path = os.path.join(output_dir, f'{experiment}_statistical_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"Full report saved to {report_path}")

    return results


def _load_method_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all metrics files grouped by method."""
    method_results = defaultdict(list)

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('_metrics.json'):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            method = data.get('method', 'unknown')
            method_results[method].append(data)

    return dict(method_results)


def _aggregate_method(runs: List[Dict]) -> Dict[str, Dict]:
    """Aggregate metrics across seeds for one method."""
    metrics_to_aggregate = ['accuracy', 'task_loss', 'functional_drift', 'cka_similarity']
    aggregated = {}

    for metric in metrics_to_aggregate:
        values = []
        for run in runs:
            steps = run.get('steps', [])
            if steps:
                # Use last 10% of steps for final performance
                n = max(1, len(steps) // 10)
                final_vals = [s.get(metric, 0) for s in steps[-n:] if metric in s]
                if final_vals:
                    values.append(np.mean(final_vals))

        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            n = len(values)
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0
            aggregated[metric] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(mean - ci),
                'ci_upper': float(mean + ci),
                'n_seeds': n,
                'raw_values': [float(v) for v in values],
            }

    # Forgetting scores
    all_forgetting = []
    for run in runs:
        fscores = run.get('forgetting_scores', [])
        if isinstance(fscores, dict):
            for _, scores in fscores.items():
                if isinstance(scores, list):
                    all_forgetting.extend(scores)
                else:
                    all_forgetting.append(scores)
        elif isinstance(fscores, list):
            all_forgetting.extend(fscores)

    if all_forgetting:
        mean_f = np.mean(all_forgetting)
        std_f = np.std(all_forgetting, ddof=1) if len(all_forgetting) > 1 else 0
        ci_f = 1.96 * std_f / np.sqrt(len(all_forgetting)) if len(all_forgetting) > 1 else 0
        aggregated['forgetting'] = {
            'mean': float(mean_f),
            'std': float(std_f),
            'ci_lower': float(mean_f - ci_f),
            'ci_upper': float(mean_f + ci_f),
            'n_values': len(all_forgetting),
        }

    return aggregated


def _compare_methods(ref_runs: List[Dict], other_runs: List[Dict], other_name: str) -> Dict:
    """Statistical comparison between reference and another method."""
    analyzer = StatisticalAnalyzer()
    metrics_to_compare = ['accuracy', 'functional_drift', 'cka_similarity', 'forgetting']
    comparison = {}

    for metric in metrics_to_compare:
        ref_vals = _extract_final_values(ref_runs, metric)
        other_vals = _extract_final_values(other_runs, metric)

        if len(ref_vals) >= 2 and len(other_vals) >= 2:
            test = analyzer.welch_t_test(ref_vals, other_vals)
            comparison[metric] = {
                'ref_mean': float(np.mean(ref_vals)),
                'ref_std': float(np.std(ref_vals, ddof=1)),
                'other_mean': float(np.mean(other_vals)),
                'other_std': float(np.std(other_vals, ddof=1)),
                't_stat': test['t_stat'],
                'p_value': test['p_value'],
                'cohens_d': test['cohens_d'],
                'significant': test['significant'],
            }

    return comparison


def _extract_final_values(runs: List[Dict], metric: str) -> List[float]:
    """Extract final metric values across seeds."""
    values = []
    if metric == 'forgetting':
        for run in runs:
            fscores = run.get('forgetting_scores', [])
            vals = []
            if isinstance(fscores, dict):
                for _, scores in fscores.items():
                    if isinstance(scores, list):
                        vals.extend(scores)
                    else:
                        vals.append(scores)
            elif isinstance(fscores, list):
                vals = fscores
            if vals:
                values.append(np.mean(vals))
    else:
        for run in runs:
            steps = run.get('steps', [])
            if steps:
                n = max(1, len(steps) // 10)
                final_vals = [s.get(metric, 0) for s in steps[-n:] if metric in s]
                if final_vals:
                    values.append(np.mean(final_vals))
    return values


def _generate_latex_table(results: Dict) -> str:
    """Generate LaTeX results table."""
    summaries = results.get('method_summaries', {})
    methods_order = ['baseline', 'weight_decay', 'ewc', 'kl_trust', 'functional_trust']
    methods = [m for m in methods_order if m in summaries]

    method_labels = {
        'baseline': 'Standard Adam',
        'weight_decay': 'Weight Decay',
        'ewc': 'EWC',
        'kl_trust': 'KL Trust Region',
        'functional_trust': r'\textbf{FTR (Ours)}',
    }

    metrics = ['accuracy', 'functional_drift', 'cka_similarity', 'forgetting']
    metric_labels = {
        'accuracy': r'Acc. $\uparrow$',
        'functional_drift': r'$D_f$ $\downarrow$',
        'cka_similarity': r'CKA $\uparrow$',
        'forgetting': r'Forget. $\downarrow$',
    }

    # Find best values
    best_vals = {}
    for metric in metrics:
        vals = {}
        for m in methods:
            if metric in summaries.get(m, {}):
                vals[m] = summaries[m][metric]['mean']
        if vals:
            if metric in ['accuracy', 'cka_similarity']:
                best_vals[metric] = max(vals.values())
            else:
                best_vals[metric] = min(vals.values())

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Main Results}')
    lines.append(r'\label{tab:main_results}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(metrics) + '}')
    lines.append(r'\toprule')
    header = 'Method & ' + ' & '.join(metric_labels.get(m, m) for m in metrics) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    for method in methods:
        label = method_labels.get(method, method)
        cells = [label]
        for metric in metrics:
            if metric in summaries.get(method, {}):
                data = summaries[method][metric]
                val = data['mean']
                std = data['std']
                is_best = abs(val - best_vals.get(metric, float('inf'))) < 1e-6
                if is_best:
                    cells.append(rf'\textbf{{{val:.3f}}} $\pm$ {std:.3f}')
                else:
                    cells.append(f'{val:.3f} $\\pm$ {std:.3f}')
            else:
                cells.append('--')
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def _generate_comparison_table(results: Dict) -> str:
    """Generate pairwise comparison table."""
    comparisons = results.get('comparisons', {})
    if not comparisons:
        return ''

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Statistical Significance (vs.\ FTR)}')
    lines.append(r'\label{tab:significance}')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\toprule')
    lines.append(r'Comparison & Metric & $t$-stat & $p$-value & Cohen\'s $d$ & Sig. \\')
    lines.append(r'\midrule')

    for comp_name, metrics in comparisons.items():
        for metric, stats in metrics.items():
            sig = "***" if stats.get('p_value', 1) < 0.001 else \
                  "**" if stats.get('p_value', 1) < 0.01 else \
                  "*" if stats.get('p_value', 1) < 0.05 else "n.s."
            lines.append(f'{comp_name} & {metric} & '
                         f'{stats.get("t_stat", 0):.3f} & '
                         f'{stats.get("p_value", 1):.4f} & '
                         f'{stats.get("cohens_d", 0):.3f} & '
                         f'{sig} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def _json_serializer(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run statistical analysis')
    parser.add_argument('--results', required=True, help='Results directory')
    parser.add_argument('--output', default='results/analysis')
    parser.add_argument('--experiment', default='continual_cifar')
    parser.add_argument('--reference', default='functional_trust')
    args = parser.parse_args()

    run_statistical_analysis(args.results, args.output, args.experiment, args.reference)
