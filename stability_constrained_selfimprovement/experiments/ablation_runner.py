# ============================================================================
# Ablation Study Runner
# Systematically varies hyperparameters and runs controlled experiments
# ============================================================================

import os
import json
import copy
import itertools
from typing import Dict, List
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.common import load_config, set_seed, get_device, ensure_dir
from experiments.exp_continual import run_continual_learning
from experiments.exp_transformer import run_transformer_experiment
from experiments.exp_rl import run_rl_experiment


def run_ablation_study(
    base_config_path: str,
    ablation_config_path: str,
    output_dir: str,
    experiment: str = 'continual_cifar',
    seeds: List[int] = None,
):
    """
    Run systematic ablation study.
    
    Varies one factor at a time while keeping others fixed.
    """
    base_config = load_config(base_config_path)
    ablation_config = load_config(ablation_config_path)

    if seeds is None:
        seeds = base_config.get('seeds', [42])

    device = get_device(base_config.get('device', 'auto'))
    ensure_dir(output_dir)

    exp_fn = {
        'continual_cifar': run_continual_learning,
        'transformer_algorithmic': run_transformer_experiment,
        'rl_gridworld': run_rl_experiment,
    }[experiment]

    results_summary = {}

    for ablation_category, variations in ablation_config.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_category}")
        print(f"{'='*60}")

        category_dir = os.path.join(output_dir, ablation_category)
        ensure_dir(category_dir)

        for variation_name, params in variations.items():
            print(f"\n  Variation: {variation_name}")
            var_dir = os.path.join(category_dir, str(variation_name))
            ensure_dir(var_dir)

            # Create modified config
            config = copy.deepcopy(base_config)
            _apply_ablation_params(config, ablation_category, params)

            for seed in seeds:
                print(f"    Seed {seed}...")
                try:
                    metrics = exp_fn(
                        method='functional_trust',
                        config=config,
                        seed=seed,
                        device=device,
                        save_dir=var_dir,
                    )
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            # Summarize this variation
            results_summary.setdefault(ablation_category, {})[variation_name] = {
                'params': params,
                'dir': var_dir,
            }

    # Save summary
    summary_path = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nAblation summary saved to {summary_path}")

    return results_summary


def _apply_ablation_params(config: Dict, category: str, params):
    """Apply ablation parameters to the config."""
    if category == 'epsilon_strategy':
        if isinstance(params, dict):
            for k, v in params.items():
                if k in config.get('epsilon_scheduler', {}):
                    config['epsilon_scheduler'][k] = v
                elif k in config.get('drift', {}):
                    config['drift'][k] = v
        else:
            config['epsilon_scheduler']['type'] = str(params)

    elif category == 'lambda_values':
        if isinstance(params, (int, float)):
            for exp_key in ['experiment_a', 'experiment_b', 'experiment_c']:
                if exp_key in config:
                    config[exp_key]['drift_lambda'] = float(params)
        elif isinstance(params, dict):
            for k, v in params.items():
                if k == 'lambda_init':
                    for exp_key in ['experiment_a', 'experiment_b', 'experiment_c']:
                        if exp_key in config:
                            config[exp_key]['drift_lambda'] = v

    elif category == 'model_size':
        if isinstance(params, dict):
            for k, v in params.items():
                if k == 'model':
                    config.setdefault('experiment_a', {})['model'] = v
        else:
            config.setdefault('experiment_a', {})['model'] = str(params)

    elif category == 'constraint_timing':
        if isinstance(params, dict):
            for k, v in params.items():
                config['drift'][k] = v

    elif category == 'no_constraint':
        # Disable constraint entirely
        for exp_key in ['experiment_a', 'experiment_b', 'experiment_c']:
            if exp_key in config:
                config[exp_key]['drift_lambda'] = 0.0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ablation', default='configs/ablation.yaml')
    parser.add_argument('--output', default='results/ablations')
    parser.add_argument('--experiment', default='continual_cifar',
                        choices=['continual_cifar', 'transformer_algorithmic', 'rl_gridworld'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    args = parser.parse_args()

    run_ablation_study(args.config, args.ablation, args.output, args.experiment, args.seeds)
