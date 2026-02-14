#!/usr/bin/env python3
"""
Ablation studies for FTR paper.

Studies:
1. Epsilon sensitivity: How does drift budget affect AA/Fgt trade-off?
2. Lambda learning rate: How does dual variable speed affect convergence?
3. Lambda momentum: Impact of EMA smoothing on stability
4. Online vs offline drift: KL on current batch vs reference set
5. Fixed vs adaptive lambda: FTR vs LwF-equivalent (fixed alpha)
6. Distillation temperature: Impact of T on drift quality
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.common import load_config, set_seed, get_device, ensure_dir
from experiments.unified_runner import run_unified_experiment


def run_ablation(name, param_grid, base_config_overrides=None, method='functional_trust',
                 benchmark='split_cifar10', seeds=[42, 137, 256], epochs=15, save_base=None):
    """Run an ablation study over a parameter grid."""
    config = load_config('configs/default.yaml')
    device = get_device('auto')
    config['experiment_a']['epochs_per_task'] = epochs
    
    if base_config_overrides:
        for key_path, value in base_config_overrides.items():
            keys = key_path.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
    
    if save_base is None:
        save_base = os.path.join('results', 'ablations', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")
    
    for param_name, param_values in param_grid.items():
        results[param_name] = {}
        
        for val in param_values:
            # Set the parameter
            keys = param_name.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val
            
            aa_list, fgt_list = [], []
            for seed in seeds:
                result = run_unified_experiment(
                    benchmark, method, config, seed=seed,
                    device=device, save_dir=f'/tmp/abl_{name}_{val}_{seed}',
                    verbose=False,
                )
                aa_list.append(result['average_accuracy'])
                fgt_list.append(result['forgetting'])
            
            results[param_name][str(val)] = {
                'aa_mean': float(np.mean(aa_list)),
                'aa_std': float(np.std(aa_list)),
                'fgt_mean': float(np.mean(fgt_list)),
                'fgt_std': float(np.std(fgt_list)),
            }
            
            print(f"  {param_name}={val}: AA={np.mean(aa_list):.4f}±{np.std(aa_list):.4f}, "
                  f"Fgt={np.mean(fgt_list):.4f}±{np.std(fgt_list):.4f}")
    
    # Save
    ensure_dir(save_base)
    with open(os.path.join(save_base, f'{name}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_all_ablations(epochs=10, seeds=[42, 137, 256]):
    """Run all ablation studies."""
    save_base = os.path.join('results', 'ablations', datetime.now().strftime('%Y%m%d_%H%M%S'))
    ensure_dir(save_base)
    
    all_results = {}
    
    # 1. Epsilon sensitivity
    print("\n" + "="*70)
    print("ABLATION 1: Epsilon Sensitivity")
    print("="*70)
    all_results['epsilon'] = run_ablation(
        'epsilon_sensitivity',
        {'experiment_a.drift_epsilon': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]},
        seeds=seeds, epochs=epochs, save_base=save_base,
    )
    
    # 2. Lambda learning rate
    print("\n" + "="*70)
    print("ABLATION 2: Lambda Learning Rate")
    print("="*70)
    all_results['lambda_lr'] = run_ablation(
        'lambda_lr',
        {'drift.lambda_lr': [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]},
        seeds=seeds, epochs=epochs, save_base=save_base,
    )
    
    # 3. Lambda momentum
    print("\n" + "="*70)
    print("ABLATION 3: Lambda Momentum")
    print("="*70)
    all_results['momentum'] = run_ablation(
        'lambda_momentum',
        {'drift.lambda_momentum': [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]},
        seeds=seeds, epochs=epochs, save_base=save_base,
    )
    
    # 4. Distillation temperature
    print("\n" + "="*70)
    print("ABLATION 4: Distillation Temperature")
    print("="*70)
    all_results['temperature'] = run_ablation(
        'temperature',
        {'lwf.temperature': [1.0, 2.0, 4.0, 8.0]},
        seeds=seeds, epochs=epochs, save_base=save_base,
    )
    
    # 5. Lambda init
    print("\n" + "="*70)
    print("ABLATION 5: Lambda Initial Value")
    print("="*70)
    all_results['lambda_init'] = run_ablation(
        'lambda_init',
        {'experiment_a.drift_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]},
        seeds=seeds, epochs=epochs, save_base=save_base,
    )
    
    # Save summary
    with open(os.path.join(save_base, 'all_ablations.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll ablations saved to: {save_base}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 137, 256])
    args = parser.parse_args()
    
    run_all_ablations(epochs=args.epochs, seeds=args.seeds)
