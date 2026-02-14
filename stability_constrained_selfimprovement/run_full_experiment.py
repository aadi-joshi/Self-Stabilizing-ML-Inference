#!/usr/bin/env python3
"""
Full experiment runner for FTR paper.
Runs all methods × all seeds × all benchmarks with proper logging.
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


def run_full_experiment(
    benchmarks=None,
    methods=None, 
    seeds=None,
    epochs=None,
    save_base=None,
):
    """Run full experiment suite."""
    config = load_config('configs/default.yaml')
    device = get_device('auto')
    
    if benchmarks is None:
        benchmarks = ['split_cifar10']
    if methods is None:
        methods = ['baseline', 'ewc', 'si', 'lwf', 'replay', 'functional_trust', 'ftr_replay']
    if seeds is None:
        seeds = [42, 137, 256]
    if epochs is None:
        epochs = 15
    if save_base is None:
        save_base = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    config['experiment_a']['epochs_per_task'] = epochs
    
    total_runs = len(benchmarks) * len(methods) * len(seeds)
    completed = 0
    start_time = time.time()
    
    all_results = {}
    
    print(f"=" * 70)
    print(f"FTR Full Experiment Suite")
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per task: {epochs}")
    print(f"Total runs: {total_runs}")
    print(f"Save directory: {save_base}")
    print(f"=" * 70)
    
    for benchmark in benchmarks:
        all_results[benchmark] = {}
        
        for method in methods:
            all_results[benchmark][method] = {'aa': [], 'fgt': [], 'bwt': []}
            
            for seed in seeds:
                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_runs - completed) if completed > 0 else 0
                
                print(f"\n[{completed}/{total_runs}] {benchmark}/{method}/seed={seed} (ETA: {eta/60:.1f}min)")
                
                save_dir = os.path.join(save_base, benchmark, method, f'seed_{seed}')
                ensure_dir(save_dir)
                
                try:
                    result = run_unified_experiment(
                        benchmark, method, config, seed=seed,
                        device=device, save_dir=save_dir, verbose=False,
                    )
                    
                    all_results[benchmark][method]['aa'].append(result['average_accuracy'])
                    all_results[benchmark][method]['fgt'].append(result['forgetting'])
                    all_results[benchmark][method]['bwt'].append(result['backward_transfer'])
                    
                    # Save individual result
                    result_file = os.path.join(save_dir, 'result.json')
                    # Convert numpy types
                    serializable = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            serializable[k] = v.tolist()
                        elif isinstance(v, (np.float32, np.float64)):
                            serializable[k] = float(v)
                        else:
                            serializable[k] = v
                    with open(result_file, 'w') as f:
                        json.dump(serializable, f, indent=2)
                    
                    print(f"  AA={result['average_accuracy']:.4f}, Fgt={result['forgetting']:.4f}")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Print aggregated results for this method
            aa = all_results[benchmark][method]['aa']
            fgt = all_results[benchmark][method]['fgt']
            if aa:
                print(f"\n  >> {method}: AA={np.mean(aa):.4f}±{np.std(aa):.4f}, Fgt={np.mean(fgt):.4f}±{np.std(fgt):.4f}")
    
    # Save aggregated results
    agg_file = os.path.join(save_base, 'aggregated_results.json')
    agg = {}
    for benchmark in all_results:
        agg[benchmark] = {}
        for method in all_results[benchmark]:
            d = all_results[benchmark][method]
            agg[benchmark][method] = {
                'average_accuracy_mean': float(np.mean(d['aa'])),
                'average_accuracy_std': float(np.std(d['aa'])),
                'forgetting_mean': float(np.mean(d['fgt'])),
                'forgetting_std': float(np.std(d['fgt'])),
                'backward_transfer_mean': float(np.mean(d['bwt'])),
                'backward_transfer_std': float(np.std(d['bwt'])),
                'n_seeds': len(d['aa']),
                'raw_aa': [float(x) for x in d['aa']],
                'raw_fgt': [float(x) for x in d['fgt']],
            }
    
    with open(agg_file, 'w') as f:
        json.dump(agg, f, indent=2)
    
    # Print final summary table
    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE ({elapsed_total/60:.1f} minutes)")
    print(f"{'=' * 70}")
    
    for benchmark in agg:
        print(f"\n{benchmark}:")
        print(f"{'Method':<25s} {'AA ↑':>12s} {'Fgt ↓':>12s} {'BWT':>12s}")
        print("-" * 65)
        for method in agg[benchmark]:
            d = agg[benchmark][method]
            print(f"{method:<25s} {d['average_accuracy_mean']:.4f}±{d['average_accuracy_std']:.4f} "
                  f"{d['forgetting_mean']:.4f}±{d['forgetting_std']:.4f} "
                  f"{d['backward_transfer_mean']:.4f}±{d['backward_transfer_std']:.4f}")
    
    print(f"\nResults saved to: {save_base}")
    return agg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', nargs='+', default=['split_cifar10'])
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 137, 256])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()
    
    run_full_experiment(
        benchmarks=args.benchmarks,
        methods=args.methods,
        seeds=args.seeds,
        epochs=args.epochs,
        save_base=args.save_dir,
    )
