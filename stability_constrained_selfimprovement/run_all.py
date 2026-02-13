#!/usr/bin/env python3
# ============================================================================
# Master Run Script
# Orchestrates all experiments, baselines, ablations, analysis, and figures
# ============================================================================
"""
Usage:
    # Full pipeline (all experiments, all seeds)
    python run_all.py

    # Single experiment
    python run_all.py --experiment continual_cifar

    # Specific methods
    python run_all.py --methods functional_trust ewc baseline

    # Quick test (1 seed, reduced epochs)
    python run_all.py --quick

    # Only generate figures from existing results
    python run_all.py --figures-only

    # Only run statistical analysis
    python run_all.py --analysis-only
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.common import load_config, set_seed, get_device, ensure_dir
from experiments.exp_continual import run_continual_learning
from experiments.exp_transformer import run_transformer_experiment
from experiments.exp_rl import run_rl_experiment
from experiments.ablation_runner import run_ablation_study
from experiments.statistical_analysis import run_statistical_analysis
from visualization.plotting import generate_all_figures


EXPERIMENTS = {
    'continual_cifar': run_continual_learning,
    'transformer_algorithmic': run_transformer_experiment,
    'rl_gridworld': run_rl_experiment,
}

ALL_METHODS = ['baseline', 'weight_decay', 'ewc', 'functional_trust']
RL_METHODS = ['baseline', 'weight_decay', 'functional_trust', 'kl_trust']


def parse_args():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--config', default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--experiment', nargs='+',
                        default=['continual_cifar', 'transformer_algorithmic', 'rl_gridworld'],
                        help='Experiments to run')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to run (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds (default: from config)')
    parser.add_argument('--output', default='results',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (1 seed, reduced epochs)')
    parser.add_argument('--figures-only', action='store_true',
                        help='Only generate figures from existing results')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Only run statistical analysis')
    parser.add_argument('--ablations', action='store_true',
                        help='Also run ablation studies')
    parser.add_argument('--ablation-config', default='configs/ablation.yaml',
                        help='Ablation config path')
    return parser.parse_args()


def run_experiment_suite(
    experiment_name: str,
    exp_fn,
    methods: list,
    config: dict,
    seeds: list,
    output_dir: str,
):
    """Run all methods × seeds for one experiment."""
    exp_dir = os.path.join(output_dir, experiment_name)
    ensure_dir(exp_dir)

    results = {}
    total = len(methods) * len(seeds)
    completed = 0

    for method in methods:
        method_results = []
        for seed in seeds:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] {experiment_name} | {method} | seed={seed}")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                metrics = exp_fn(
                    method=method,
                    config=config,
                    seed=seed,
                    device=get_device(config.get('device', 'auto')),
                    save_dir=exp_dir,
                )
                elapsed = time.time() - t0
                print(f"  Completed in {elapsed:.1f}s")
                method_results.append({'seed': seed, 'status': 'success', 'time': elapsed})
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED after {elapsed:.1f}s: {e}")
                import traceback
                traceback.print_exc()
                method_results.append({'seed': seed, 'status': 'failed', 'error': str(e)})

        results[method] = method_results

    # Save run log
    log_path = os.path.join(exp_dir, 'run_log.json')
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def apply_quick_settings(config: dict) -> dict:
    """Reduce settings for quick testing."""
    import copy
    config = copy.deepcopy(config)

    # Reduce epochs
    for exp_key in ['experiment_a', 'experiment_b', 'experiment_c']:
        if exp_key in config:
            if 'epochs_per_task' in config[exp_key]:
                config[exp_key]['epochs_per_task'] = 3
            if 'episodes' in config[exp_key]:
                config[exp_key]['episodes'] = 200

    # Reduce reference points
    config['drift']['num_reference_points'] = 100

    return config


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"{'#'*60}")
    print(f"# Stability-Constrained Self-Improving Agents")
    print(f"# Functional Trust Regions — Full Experiment Pipeline")
    print(f"# Timestamp: {timestamp}")
    print(f"{'#'*60}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    seeds = args.seeds or config.get('seeds', [42, 137, 256, 512, 1024])
    output_dir = os.path.join(os.path.dirname(__file__), args.output, timestamp)
    ensure_dir(output_dir)

    if args.quick:
        config = apply_quick_settings(config)
        seeds = seeds[:1]
        print("  ** QUICK MODE: 1 seed, reduced epochs **")

    # Save config snapshot
    config_snapshot = os.path.join(output_dir, 'config_snapshot.json')
    with open(config_snapshot, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # --- Run experiments ---
    if not args.figures_only and not args.analysis_only:
        for exp_name in args.experiment:
            if exp_name not in EXPERIMENTS:
                print(f"Unknown experiment: {exp_name}")
                continue

            if args.methods:
                methods = args.methods
            elif exp_name == 'rl_gridworld':
                methods = RL_METHODS
            else:
                methods = ALL_METHODS

            run_experiment_suite(
                experiment_name=exp_name,
                exp_fn=EXPERIMENTS[exp_name],
                methods=methods,
                config=config,
                seeds=seeds,
                output_dir=output_dir,
            )

    # --- Ablations ---
    if args.ablations and not args.figures_only and not args.analysis_only:
        ablation_config = os.path.join(os.path.dirname(__file__), args.ablation_config)
        if os.path.exists(ablation_config):
            for exp_name in args.experiment:
                run_ablation_study(
                    base_config_path=config_path,
                    ablation_config_path=ablation_config,
                    output_dir=os.path.join(output_dir, 'ablations', exp_name),
                    experiment=exp_name,
                    seeds=seeds[:2],  # Fewer seeds for ablations
                )

    # --- Statistical Analysis ---
    if not args.figures_only:
        for exp_name in args.experiment:
            exp_dir = os.path.join(output_dir, exp_name)
            if os.path.exists(exp_dir):
                analysis_dir = os.path.join(output_dir, 'analysis', exp_name)
                run_statistical_analysis(
                    results_dir=exp_dir,
                    output_dir=analysis_dir,
                    experiment=exp_name,
                )

    # --- Figures ---
    for exp_name in args.experiment:
        exp_dir = os.path.join(output_dir, exp_name)
        if os.path.exists(exp_dir):
            figures_dir = os.path.join(output_dir, 'figures', exp_name)
            generate_all_figures(
                results_dir=exp_dir,
                output_dir=figures_dir,
                experiment=exp_name,
            )

    print(f"\n{'#'*60}")
    print(f"# All done! Results in: {output_dir}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
