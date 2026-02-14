#!/usr/bin/env python3
# ============================================================================
# FTR Research — Master Experiment Pipeline
# 
# Runs the complete experimental protocol:
#   Phase 1: Core experiments (all benchmarks × all methods × 5 seeds)
#   Phase 2: Ablation studies
#   Phase 3: Statistical analysis
#   Phase 4: Figure generation
#   Phase 5: Results compilation
#
# Usage:
#   python run_ftr_experiments.py                    # Full pipeline
#   python run_ftr_experiments.py --quick             # Quick test (1 seed, 3 epochs)
#   python run_ftr_experiments.py --benchmarks split_cifar10 permuted_mnist
#   python run_ftr_experiments.py --methods baseline ewc functional_trust
#   python run_ftr_experiments.py --figures-only --results-dir results/XXXXXXXX
# ============================================================================

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from utils.common import load_config, set_seed, get_device, ensure_dir
from experiments.unified_runner import (
    run_unified_experiment, run_full_benchmark_suite,
    ALL_METHODS, ALL_BENCHMARKS, _aggregate_results,
)
from experiments.exp_transformer import run_transformer_experiment
from experiments.exp_rl import run_rl_experiment
from experiments.statistical_analysis import run_statistical_analysis
from visualization.publication_plots import generate_all_publication_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description='FTR Research — Complete Experiment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ftr_experiments.py --quick
  python run_ftr_experiments.py --benchmarks split_cifar10 --methods baseline ewc functional_trust
  python run_ftr_experiments.py --ablations
  python run_ftr_experiments.py --figures-only --results-dir results/20260213_120000
        """
    )
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                        help=f'Benchmarks to run. Options: {ALL_BENCHMARKS}')
    parser.add_argument('--methods', nargs='+', default=None,
                        help=f'Methods to run. Options: {ALL_METHODS}')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--output', default='results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 1 seed, 3 epochs per task')
    parser.add_argument('--medium', action='store_true',
                        help='Medium mode: 3 seeds, 10 epochs per task')
    parser.add_argument('--ablations', action='store_true',
                        help='Run ablation studies')
    parser.add_argument('--figures-only', action='store_true')
    parser.add_argument('--results-dir', default=None,
                        help='Existing results directory (for --figures-only)')
    parser.add_argument('--no-transformer', action='store_true',
                        help='Skip transformer experiment')
    parser.add_argument('--no-rl', action='store_true',
                        help='Skip RL experiment')
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("  Functional Trust Regions (FTR)")
    print("  Stability-Constrained Continual Learning")
    print(f"  Timestamp: {timestamp}")
    print("=" * 70)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    # Determine seeds
    seeds = args.seeds or config.get('seeds', [42, 137, 256, 512, 1024])

    # Determine benchmarks and methods
    benchmarks = args.benchmarks or ['split_cifar10', 'permuted_mnist', 'rotated_mnist']
    methods = args.methods or ['baseline', 'ewc', 'si', 'lwf',
                                'replay', 'functional_trust', 'ftr_replay']

    # Apply quick/medium settings
    if args.quick:
        seeds = seeds[:1]
        config['experiment_a']['epochs_per_task'] = 3
        config['drift']['num_reference_points'] = 100
        print("  ** QUICK MODE: 1 seed, 3 epochs **")
    elif args.medium:
        seeds = seeds[:3]
        config['experiment_a']['epochs_per_task'] = 10
        print("  ** MEDIUM MODE: 3 seeds, 10 epochs **")

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.output, timestamp)
    if args.figures_only and args.results_dir:
        output_dir = args.results_dir
    ensure_dir(output_dir)

    device = get_device(config.get('device', 'auto'))
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Methods: {methods}")
    print(f"  Output: {output_dir}")

    # Save config snapshot
    config_path_save = os.path.join(output_dir, 'config_snapshot.json')
    with open(config_path_save, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # ====================================================================
    # Phase 1: Core Experiments
    # ====================================================================
    raw_results = defaultdict(lambda: defaultdict(list))

    if not args.figures_only:
        print("\n" + "=" * 70)
        print("  PHASE 1: Core Continual Learning Experiments")
        print("=" * 70)

        total_runs = len(benchmarks) * len(methods) * len(seeds)
        completed = 0

        for benchmark in benchmarks:
            for method in methods:
                for seed in seeds:
                    completed += 1
                    print(f"\n{'─'*60}")
                    print(f"[{completed}/{total_runs}] {benchmark} | {method} | seed={seed}")
                    print(f"{'─'*60}")

                    t0 = time.time()
                    try:
                        result = run_unified_experiment(
                            benchmark=benchmark,
                            method=method,
                            config=config,
                            seed=seed,
                            device=device,
                            save_dir=os.path.join(output_dir, benchmark),
                            verbose=args.verbose,
                        )
                        raw_results[benchmark][method].append(result)
                        elapsed = time.time() - t0
                        print(f"  ✓ Completed in {elapsed:.1f}s")
                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"  ✗ FAILED after {elapsed:.1f}s: {e}")
                        import traceback
                        traceback.print_exc()

        # --- Transformer experiment ---
        if not args.no_transformer:
            print("\n" + "=" * 70)
            print("  Transformer Algorithmic Experiment")
            print("=" * 70)
            
            transformer_methods = ['baseline', 'weight_decay', 'ewc', 'functional_trust']
            for method in transformer_methods:
                if method not in methods and method != 'functional_trust':
                    continue
                for seed in seeds:
                    print(f"\n  Transformer | {method} | seed={seed}")
                    try:
                        run_transformer_experiment(
                            method=method, config=config, seed=seed,
                            device=device,
                            save_dir=os.path.join(output_dir, 'transformer'),
                        )
                    except Exception as e:
                        print(f"  Failed: {e}")

        # --- RL experiment ---
        if not args.no_rl:
            print("\n" + "=" * 70)
            print("  RL Gridworld Experiment")
            print("=" * 70)
            
            rl_methods = ['baseline', 'weight_decay', 'functional_trust', 'kl_trust']
            for method in rl_methods:
                for seed in seeds:
                    print(f"\n  RL | {method} | seed={seed}")
                    try:
                        run_rl_experiment(
                            method=method, config=config, seed=seed,
                            device=device,
                            save_dir=os.path.join(output_dir, 'rl_gridworld'),
                        )
                    except Exception as e:
                        print(f"  Failed: {e}")

    # ====================================================================
    # Phase 2: Ablation Studies
    # ====================================================================
    if args.ablations and not args.figures_only:
        print("\n" + "=" * 70)
        print("  PHASE 2: Ablation Studies")
        print("=" * 70)
        
        ablation_benchmark = 'split_cifar10'
        ablation_dir = os.path.join(output_dir, 'ablations')
        ensure_dir(ablation_dir)
        ablation_seeds = seeds[:2]  # Fewer seeds for ablations

        # Ablation 1: Epsilon schedules
        print("\n  Ablation: Epsilon Schedules")
        eps_configs = {
            'fixed_low': {'type': 'fixed', 'epsilon_init': 0.1},
            'fixed_mid': {'type': 'fixed', 'epsilon_init': 1.0},
            'fixed_high': {'type': 'fixed', 'epsilon_init': 5.0},
            'cosine': {'type': 'cosine', 'epsilon_init': 1.0},
            'uncertainty': {'type': 'uncertainty', 'epsilon_init': 1.0},
        }
        for name, eps_cfg in eps_configs.items():
            cfg = _deep_copy_config(config)
            cfg['epsilon_scheduler']['type'] = eps_cfg['type']
            cfg['experiment_a']['drift_epsilon'] = eps_cfg['epsilon_init']
            for seed in ablation_seeds:
                try:
                    run_unified_experiment(
                        benchmark=ablation_benchmark, method='functional_trust',
                        config=cfg, seed=seed, device=device,
                        save_dir=os.path.join(ablation_dir, 'epsilon_schedule', name),
                    )
                except Exception as e:
                    print(f"    Failed: {e}")

        # Ablation 2: Lambda sensitivity
        print("\n  Ablation: Lambda Sensitivity")
        for lam in [0.01, 0.1, 1.0, 10.0]:
            cfg = _deep_copy_config(config)
            cfg['experiment_a']['drift_lambda'] = lam
            for seed in ablation_seeds:
                try:
                    run_unified_experiment(
                        benchmark=ablation_benchmark, method='functional_trust',
                        config=cfg, seed=seed, device=device,
                        save_dir=os.path.join(ablation_dir, 'lambda', f'lambda_{lam}'),
                    )
                except Exception as e:
                    print(f"    Failed: {e}")

        # Ablation 3: Output-space vs feature-space
        print("\n  Ablation: Output vs Feature Space Constraint")
        for constraint_type in ['functional_trust', 'feature_trust']:
            for seed in ablation_seeds:
                try:
                    run_unified_experiment(
                        benchmark=ablation_benchmark, method=constraint_type,
                        config=config, seed=seed, device=device,
                        save_dir=os.path.join(ablation_dir, 'constraint_space', constraint_type),
                    )
                except Exception as e:
                    print(f"    Failed: {e}")

    # ====================================================================
    # Phase 3: Aggregate and Analyze
    # ====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Statistical Analysis")
    print("=" * 70)

    # Load raw results if running figures-only
    if args.figures_only:
        raw_results = _load_raw_results(output_dir, benchmarks, methods)

    # Aggregate results
    aggregated = {}
    for benchmark in benchmarks:
        aggregated[benchmark] = {}
        for method in methods:
            results_list = raw_results[benchmark][method]
            if results_list:
                aggregated[benchmark][method] = _aggregate_results(results_list)

    # Save aggregated
    agg_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary table
    _print_summary_table(aggregated, benchmarks, methods)

    # Statistical significance tests
    _run_significance_tests(aggregated, benchmarks, methods, output_dir)

    # ====================================================================
    # Phase 4: Figure Generation
    # ====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: Publication-Quality Figures")
    print("=" * 70)

    figures_dir = os.path.join(output_dir, 'figures')
    try:
        generate_all_publication_figures(
            aggregated_results=aggregated,
            raw_results=dict(raw_results),
            save_dir=figures_dir,
        )
    except Exception as e:
        print(f"  Figure generation error: {e}")
        import traceback
        traceback.print_exc()

    # ====================================================================
    # Phase 5: Results Compilation
    # ====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 5: Results Compilation")
    print("=" * 70)

    _generate_latex_tables(aggregated, benchmarks, methods, output_dir)

    print(f"\n{'='*70}")
    print(f"  All done! Results saved to: {output_dir}")
    print(f"{'='*70}")


def _deep_copy_config(config):
    import copy
    return copy.deepcopy(config)


def _load_raw_results(results_dir, benchmarks, methods):
    """Load saved results from disk."""
    raw_results = defaultdict(lambda: defaultdict(list))
    
    for benchmark in benchmarks:
        bench_dir = os.path.join(results_dir, benchmark)
        if not os.path.exists(bench_dir):
            continue
        for fname in sorted(os.listdir(bench_dir)):
            if fname.endswith('.json'):
                fpath = os.path.join(bench_dir, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    method = data.get('method', '')
                    if method in methods:
                        raw_results[benchmark][method].append(data)
                except Exception:
                    pass

    return raw_results


METHOD_ORDER = [
    'baseline', 'weight_decay', 'ewc', 'si', 'lwf',
    'distillation', 'replay', 'feature_trust', 'functional_trust', 'ftr_replay',
]

def _print_summary_table(aggregated, benchmarks, methods):
    """Print a formatted summary table to stdout."""
    method_order = [m for m in METHOD_ORDER if m in methods]
    
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY (Average Accuracy ± Std)")
    print("=" * 90)
    
    header = f"{'Method':<22}"
    for b in benchmarks:
        header += f" | {b:<20}"
    print(header)
    print("-" * 90)
    
    for method in method_order:
        row = f"{method:<22}"
        for benchmark in benchmarks:
            mdata = aggregated.get(benchmark, {}).get(method, {})
            acc = mdata.get('average_accuracy', {})
            if acc:
                row += f" | {acc.get('mean', 0):.4f}±{acc.get('std', 0):.4f}      "
            else:
                row += f" | {'—':<20}"
        print(row)

    print("\n" + "=" * 90)
    print("  FORGETTING (↓ better)")
    print("=" * 90)
    
    for method in method_order:
        row = f"{method:<22}"
        for benchmark in benchmarks:
            mdata = aggregated.get(benchmark, {}).get(method, {})
            fgt = mdata.get('forgetting', {})
            if fgt:
                row += f" | {fgt.get('mean', 0):.4f}±{fgt.get('std', 0):.4f}      "
            else:
                row += f" | {'—':<20}"
        print(row)


def _run_significance_tests(aggregated, benchmarks, methods, output_dir):
    """Run Welch's t-test comparing FTR against all baselines."""
    from scipy import stats
    
    print("\n  Statistical Significance Tests (FTR vs Others)")
    print("  " + "-" * 60)
    
    ref_method = 'functional_trust'
    significance_results = {}
    
    for benchmark in benchmarks:
        bench_data = aggregated.get(benchmark, {})
        ref_data = bench_data.get(ref_method, {})
        ref_acc = ref_data.get('average_accuracy', {}).get('values', [])
        
        if len(ref_acc) < 2:
            continue

        significance_results[benchmark] = {}
        
        for method in methods:
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
            
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            print(f"  {benchmark} | FTR vs {method}: "
                  f"t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} {sig}")
            
            significance_results[benchmark][method] = {
                't_stat': float(t_stat), 'p_value': float(p_val),
                'cohens_d': float(d), 'significant': bool(p_val < 0.05),
            }

    # Save
    sig_path = os.path.join(output_dir, 'significance_tests.json')
    with open(sig_path, 'w') as f:
        json.dump(significance_results, f, indent=2)


def _generate_latex_tables(aggregated, benchmarks, methods, output_dir):
    """Generate LaTeX tables for the paper."""
    method_order = [m for m in METHOD_ORDER if m in methods]
    method_labels = {
        'baseline': 'Vanilla Fine-tuning',
        'weight_decay': 'Weight Decay',
        'ewc': 'EWC',
        'si': 'SI',
        'lwf': 'LwF',
        'distillation': 'Fixed Distillation',
        'replay': 'Replay',
        'functional_trust': r'\textbf{FTR (Ours)}',
        'feature_trust': 'Feature Trust',
        'ftr_replay': r'\textbf{FTR + Replay (Ours)}',
    }

    for benchmark in benchmarks:
        bench_data = aggregated.get(benchmark, {})
        if not bench_data:
            continue

        lines = []
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering')
        lines.append(r'\caption{Results on ' + benchmark.replace('_', ' ').title() + 
                     r' (mean $\pm$ std across 5 seeds).}')
        lines.append(r'\label{tab:' + benchmark + r'}')
        lines.append(r'\begin{tabular}{lcccc}')
        lines.append(r'\toprule')
        lines.append(r'Method & Avg. Acc. $\uparrow$ & Forgetting $\downarrow$ & BWT $\uparrow$ & FWT $\uparrow$ \\')
        lines.append(r'\midrule')

        # Find best values for bolding
        metrics_list = ['average_accuracy', 'forgetting', 'backward_transfer', 'forward_transfer']
        best = {}
        for m in metrics_list:
            vals = {}
            for method in method_order:
                mdata = bench_data.get(method, {}).get(m, {})
                if mdata:
                    vals[method] = mdata.get('mean', 0)
            if vals:
                if m in ['average_accuracy', 'backward_transfer', 'forward_transfer']:
                    best[m] = max(vals.values())
                else:
                    best[m] = min(vals.values())

        for method in method_order:
            mdata = bench_data.get(method, {})
            label = method_labels.get(method, method)
            cells = [label]
            
            for metric in metrics_list:
                metric_data = mdata.get(metric, {})
                if metric_data:
                    mean = metric_data.get('mean', 0)
                    std = metric_data.get('std', 0)
                    is_best = abs(mean - best.get(metric, float('inf'))) < 1e-6
                    if is_best:
                        cells.append(rf'\textbf{{{mean:.3f}}} $\pm$ {std:.3f}')
                    else:
                        cells.append(f'{mean:.3f} $\\pm$ {std:.3f}')
                else:
                    cells.append('--')
            
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        table_path = os.path.join(output_dir, f'{benchmark}_results.tex')
        with open(table_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  LaTeX table saved: {table_path}")


if __name__ == '__main__':
    main()
