#!/usr/bin/env python3
"""
FTR Results Analysis Script
============================
Standalone script to analyze experiment results and generate all figures.

Usage:
    python analyze_results.py --results-dir results/20260213_XXXXXX
    python analyze_results.py --results-dir results/20260213_XXXXXX --figures-only
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from visualization.publication_plots import (
    generate_all_publication_figures,
    setup_style,
    COLORS, METHOD_LABELS, METHOD_ORDER, METHOD_MARKERS,
    save_figure,
)


def load_all_results(results_dir: str) -> Dict:
    """Load all individual experiment JSON files from a results directory."""
    raw_results = defaultdict(lambda: defaultdict(list))
    
    for item in sorted(os.listdir(results_dir)):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item not in ('figures', 'ablations'):
            # This is a benchmark directory
            benchmark = item
            for fname in sorted(os.listdir(item_path)):
                if fname.endswith('.json'):
                    fpath = os.path.join(item_path, fname)
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        method = data.get('method', '')
                        raw_results[benchmark][method].append(data)
                    except Exception as e:
                        print(f"  Warning: Could not load {fpath}: {e}")
    
    return dict(raw_results)


def aggregate_results(raw_results: Dict) -> Dict:
    """Aggregate results across seeds for each benchmark × method."""
    aggregated = {}
    
    for benchmark, methods_data in raw_results.items():
        aggregated[benchmark] = {}
        for method, results_list in methods_data.items():
            if not results_list:
                continue
            
            metrics = ['average_accuracy', 'backward_transfer', 'forward_transfer', 'forgetting']
            agg = {}
            
            for m in metrics:
                values = [r[m] for r in results_list if m in r]
                if values:
                    agg[m] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        'ci95': float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                        'values': values,
                        'n_seeds': len(values),
                    }
            
            # Aggregate accuracy matrices
            matrices = [np.array(r['accuracy_matrix']) for r in results_list if 'accuracy_matrix' in r]
            if matrices:
                stacked = np.stack(matrices)
                agg['accuracy_matrix_mean'] = stacked.mean(axis=0).tolist()
                if len(matrices) > 1:
                    agg['accuracy_matrix_std'] = stacked.std(axis=0, ddof=1).tolist()
                else:
                    agg['accuracy_matrix_std'] = np.zeros_like(matrices[0]).tolist()
            
            aggregated[benchmark][method] = agg
    
    return aggregated


def print_summary_tables(aggregated: Dict):
    """Print comprehensive summary tables to stdout."""
    for benchmark, bench_data in aggregated.items():
        methods = [m for m in METHOD_ORDER if m in bench_data]
        if not methods:
            continue
        
        print(f"\n{'='*100}")
        print(f"  {benchmark.replace('_', ' ').upper()}")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Method':<24} | {'Avg. Acc. ↑':>16} | {'Forgetting ↓':>16} | {'BWT ↑':>16} | {'FWT ↑':>16} | {'N':>3}"
        print(header)
        print("-" * 100)
        
        # Find best values
        best = {}
        for metric in ['average_accuracy', 'forgetting', 'backward_transfer', 'forward_transfer']:
            vals = {}
            for method in methods:
                mdata = bench_data.get(method, {}).get(metric, {})
                if mdata:
                    vals[method] = mdata.get('mean', 0)
            if vals:
                if metric in ['average_accuracy', 'backward_transfer', 'forward_transfer']:
                    best[metric] = max(vals.values())
                else:
                    best[metric] = min(vals.values())
        
        for method in methods:
            mdata = bench_data.get(method, {})
            label = METHOD_LABELS.get(method, method)
            
            cells = [f"{label:<24}"]
            n_seeds = 0
            for metric in ['average_accuracy', 'forgetting', 'backward_transfer', 'forward_transfer']:
                metric_data = mdata.get(metric, {})
                if metric_data:
                    mean = metric_data.get('mean', 0)
                    std = metric_data.get('std', 0)
                    n_seeds = metric_data.get('n_seeds', 0)
                    is_best = abs(mean - best.get(metric, float('inf'))) < 1e-6
                    marker = '★' if is_best else ' '
                    cells.append(f"{marker}{mean:.4f}±{std:.4f}")
                else:
                    cells.append(f"{'—':>16}")
            cells.append(f"{n_seeds:>3}")
            
            print(" | ".join(cells))
        
        print()


def run_significance_tests(aggregated: Dict):
    """Run statistical significance tests: FTR vs all baselines."""
    from scipy import stats
    
    print("\n" + "=" * 80)
    print("  STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    for benchmark, bench_data in aggregated.items():
        print(f"\n  --- {benchmark} ---")
        
        # Test FTR vs all baselines
        for ref_method in ['functional_trust', 'ftr_replay']:
            ref_data = bench_data.get(ref_method, {})
            ref_acc = ref_data.get('average_accuracy', {}).get('values', [])
            
            if len(ref_acc) < 2:
                continue
            
            ref_label = METHOD_LABELS.get(ref_method, ref_method)
            print(f"\n  {ref_label} vs baselines:")
            
            for method in METHOD_ORDER:
                if method == ref_method:
                    continue
                mdata = bench_data.get(method, {})
                m_acc = mdata.get('average_accuracy', {}).get('values', [])
                
                if len(m_acc) < 2:
                    continue
                
                t_stat, p_val = stats.ttest_ind(ref_acc, m_acc, equal_var=False)
                
                # Cohen's d
                pooled_std = np.sqrt((np.std(ref_acc, ddof=1)**2 + np.std(m_acc, ddof=1)**2) / 2)
                d = (np.mean(ref_acc) - np.mean(m_acc)) / max(pooled_std, 1e-10)
                
                # Direction
                direction = "better" if np.mean(ref_acc) > np.mean(m_acc) else "worse"
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                
                m_label = METHOD_LABELS.get(method, method)
                print(f"    vs {m_label:<24}: Δ={np.mean(ref_acc)-np.mean(m_acc):+.4f} "
                      f"({direction}), t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} {sig}")


def compute_improvement_table(aggregated: Dict):
    """Show percentage improvement of FTR over each baseline."""
    print("\n" + "=" * 80)
    print("  RELATIVE IMPROVEMENT OF FTR OVER BASELINES")
    print("=" * 80)
    
    for benchmark, bench_data in aggregated.items():
        print(f"\n  --- {benchmark} ---")
        
        for ref_method in ['functional_trust', 'ftr_replay']:
            ref_data = bench_data.get(ref_method, {})
            ref_acc = ref_data.get('average_accuracy', {}).get('mean', 0)
            ref_fgt = ref_data.get('forgetting', {}).get('mean', 0)
            
            if ref_acc == 0:
                continue
            
            ref_label = METHOD_LABELS.get(ref_method, ref_method)
            print(f"\n  {ref_label} (AA={ref_acc:.4f}, Fgt={ref_fgt:.4f}):")
            
            for method in METHOD_ORDER:
                if method == ref_method:
                    continue
                mdata = bench_data.get(method, {})
                m_acc = mdata.get('average_accuracy', {}).get('mean', 0)
                m_fgt = mdata.get('forgetting', {}).get('mean', 0)
                
                if m_acc == 0:
                    continue
                
                acc_improv = ((ref_acc - m_acc) / max(m_acc, 1e-10)) * 100
                fgt_improv = ((m_fgt - ref_fgt) / max(m_fgt, 1e-10)) * 100 if m_fgt > 0 else 0
                
                m_label = METHOD_LABELS.get(method, method)
                print(f"    vs {m_label:<24}: "
                      f"Acc: {acc_improv:+.1f}%, Forgetting reduction: {fgt_improv:+.1f}%")


def generate_latex_tables(aggregated: Dict, output_dir: str):
    """Generate LaTeX tables ready for paper inclusion."""
    method_order = [m for m in METHOD_ORDER if any(m in bench_data for bench_data in aggregated.values())]
    
    method_labels = {
        'baseline': 'Fine-tuning',
        'weight_decay': 'Weight Decay',
        'ewc': 'EWC \\citep{kirkpatrick2017overcoming}',
        'si': 'SI \\citep{zenke2017continual}',
        'lwf': 'LwF \\citep{li2016learning}',
        'distillation': 'Fixed Distillation',
        'replay': 'ER \\citep{chaudhry2019continual}',
        'functional_trust': '\\textbf{FTR (Ours)}',
        'feature_trust': 'Feature Trust',
        'ftr_replay': '\\textbf{FTR+ER (Ours)}',
    }
    
    benchmarks = list(aggregated.keys())
    
    # Generate one comprehensive table
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Continual learning results across benchmarks. ')
    lines.append(r'We report Average Accuracy (AA $\uparrow$) and Forgetting (F $\downarrow$) ')
    lines.append(r'averaged over 3 seeds $\pm$ standard deviation. ')
    lines.append(r'\textbf{Bold}: best result. \underline{Underline}: second best.}')
    lines.append(r'\label{tab:main_results}')
    lines.append(r'\small')
    
    # Build column spec
    n_bench = len(benchmarks)
    col_spec = 'l' + 'cc' * n_bench
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    
    # Header row 1: benchmark names
    header1 = 'Method'
    for b in benchmarks:
        b_name = b.replace('_', ' ').title()
        header1 += f' & \\multicolumn{{2}}{{c}}{{{b_name}}}'
    header1 += r' \\'
    lines.append(header1)
    
    # Header row 2: metrics
    header2 = ''
    for b in benchmarks:
        header2 += r' & AA $\uparrow$ & F $\downarrow$'
    header2 += r' \\'
    lines.append(r'\cmidrule(lr){2-' + str(1 + 2*n_bench) + '}')
    lines.append(header2)
    lines.append(r'\midrule')
    
    # Find best and second-best per metric per benchmark
    for method in method_order:
        label = method_labels.get(method, method)
        row = label
        
        for benchmark in benchmarks:
            bench_data = aggregated.get(benchmark, {})
            mdata = bench_data.get(method, {})
            
            for metric in ['average_accuracy', 'forgetting']:
                metric_data = mdata.get(metric, {})
                if metric_data:
                    mean = metric_data.get('mean', 0)
                    std = metric_data.get('std', 0)
                    
                    # Check if best
                    all_vals = {m: bench_data.get(m, {}).get(metric, {}).get('mean', float('inf') if metric == 'forgetting' else 0) 
                               for m in method_order if m in bench_data}
                    if metric in ['average_accuracy']:
                        sorted_vals = sorted(all_vals.values(), reverse=True)
                    else:
                        sorted_vals = sorted(all_vals.values())
                    
                    is_best = len(sorted_vals) > 0 and abs(mean - sorted_vals[0]) < 1e-6
                    is_second = len(sorted_vals) > 1 and abs(mean - sorted_vals[1]) < 1e-6 and not is_best
                    
                    val_str = f'{mean:.3f}$\\pm${std:.3f}'
                    if is_best:
                        val_str = f'\\textbf{{{mean:.3f}}}$\\pm${std:.3f}'
                    elif is_second:
                        val_str = f'\\underline{{{mean:.3f}}}$\\pm${std:.3f}'
                    
                    row += f' & {val_str}'
                else:
                    row += ' & --'
        
        row += r' \\'
        
        # Add midrule before our methods
        if method == 'replay' or method == 'feature_trust':
            if method == list(method_order)[-3]:  # Before our methods
                lines.append(r'\midrule')
        
        lines.append(row)
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    
    table_path = os.path.join(output_dir, 'main_results_table.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved: {table_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze FTR experiment results')
    parser.add_argument('--results-dir', required=True, help='Results directory to analyze')
    parser.add_argument('--figures-only', action='store_true', help='Only generate figures')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    
    print(f"Loading results from: {results_dir}")
    
    # Load results
    raw_results = load_all_results(results_dir)
    if not raw_results:
        print("No results found!")
        return
    
    # Count
    total = sum(len(v) for methods in raw_results.values() for v in methods.values())
    print(f"Loaded {total} individual experiment results")
    for benchmark, methods in raw_results.items():
        for method, results in methods.items():
            print(f"  {benchmark} | {method}: {len(results)} seeds")
    
    # Aggregate
    aggregated = aggregate_results(raw_results)
    
    # Save aggregated
    agg_path = os.path.join(results_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to: {agg_path}")
    
    if not args.figures_only:
        # Print tables
        print_summary_tables(aggregated)
        
        # Statistical tests
        try:
            run_significance_tests(aggregated)
        except ImportError:
            print("  (scipy not available for significance tests)")
        
        # Improvement analysis
        compute_improvement_table(aggregated)
        
        # LaTeX tables
        generate_latex_tables(aggregated, results_dir)
    
    # Generate figures
    if not args.no_figures:
        print("\n" + "=" * 80)
        print("  GENERATING FIGURES")
        print("=" * 80)
        
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        try:
            generate_all_publication_figures(
                aggregated_results=aggregated,
                raw_results=raw_results,
                save_dir=figures_dir,
            )
            print(f"\n  Figures saved to: {figures_dir}")
        except Exception as e:
            print(f"  Figure generation error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
