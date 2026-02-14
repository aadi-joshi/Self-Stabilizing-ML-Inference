#!/usr/bin/env python3
"""
Lean experiment runner — runs each experiment sequentially and saves results.
Designed to be robust (saves after each run) and fast.
"""

import os, sys, json, time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from utils.common import load_config, set_seed, get_device, ensure_dir
from experiments.unified_runner import run_unified_experiment, _aggregate_results


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = load_config(os.path.join(os.path.dirname(__file__), 'configs/default.yaml'))
    
    # Configuration
    seeds = [42, 137, 256]
    benchmarks = ['split_cifar10']
    methods = ['baseline', 'ewc', 'si', 'lwf', 'replay', 'functional_trust', 'ftr_replay']
    epochs = 15
    
    config['experiment_a']['epochs_per_task'] = epochs
    config['experiment_a']['model'] = 'resnet18_small'  # Fast
    
    device = get_device('auto')
    output_dir = os.path.join(os.path.dirname(__file__), 'results', timestamp)
    ensure_dir(output_dir)
    
    # Save config
    with open(os.path.join(output_dir, 'config_snapshot.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    total_runs = len(benchmarks) * len(methods) * len(seeds)
    print(f"{'='*70}")
    print(f"FTR Experiments | {timestamp}")
    print(f"Seeds: {seeds} | Epochs: {epochs} | Total runs: {total_runs}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    raw_results = defaultdict(lambda: defaultdict(list))
    completed = 0
    total_time = 0
    
    for benchmark in benchmarks:
        for method in methods:
            for seed in seeds:
                completed += 1
                print(f"\n[{completed}/{total_runs}] {benchmark} | {method} | seed={seed}")
                t0 = time.time()
                
                try:
                    result = run_unified_experiment(
                        benchmark=benchmark,
                        method=method,
                        config=config,
                        seed=seed,
                        device=device,
                        save_dir=os.path.join(output_dir, benchmark),
                        verbose=True,
                    )
                    raw_results[benchmark][method].append(result)
                    elapsed = time.time() - t0
                    total_time += elapsed
                    
                    aa = result['average_accuracy']
                    fgt = result['forgetting']
                    print(f"  ✓ AA={aa:.4f}, Fgt={fgt:.4f} ({elapsed:.0f}s)")
                    
                    # Estimate remaining time
                    avg_per_run = total_time / completed
                    remaining = avg_per_run * (total_runs - completed)
                    print(f"  ETA: {remaining/60:.0f} min remaining")
                    
                except Exception as e:
                    elapsed = time.time() - t0
                    total_time += elapsed
                    print(f"  ✗ FAILED: {e} ({elapsed:.0f}s)")
                    import traceback
                    traceback.print_exc()
    
    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}")
    
    aggregated = {}
    for benchmark in benchmarks:
        aggregated[benchmark] = {}
        for method in methods:
            results_list = raw_results[benchmark][method]
            if results_list:
                aggregated[benchmark][method] = _aggregate_results(results_list)
    
    # Save
    agg_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary
    print(f"\n{'='*90}")
    print("RESULTS SUMMARY")
    print(f"{'='*90}")
    
    method_order = ['baseline', 'ewc', 'si', 'lwf', 'replay', 'functional_trust', 'ftr_replay']
    
    header = f"{'Method':<22} | {'Avg Acc ↑':>14} | {'Forgetting ↓':>14} | {'BWT ↑':>14}"
    print(header)
    print("-" * 72)
    
    for method in method_order:
        for benchmark in benchmarks:
            mdata = aggregated.get(benchmark, {}).get(method, {})
            aa = mdata.get('average_accuracy', {})
            fgt = mdata.get('forgetting', {})
            bwt = mdata.get('backward_transfer', {})
            
            if aa:
                row = f"{method:<22} | {aa['mean']:.4f}±{aa['std']:.4f}   | {fgt.get('mean',0):.4f}±{fgt.get('std',0):.4f}   | {bwt.get('mean',0):.4f}±{bwt.get('std',0):.4f}"
                print(row)
    
    # Generate figures
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")
    
    figures_dir = os.path.join(output_dir, 'figures')
    try:
        from visualization.publication_plots import generate_all_publication_figures
        generate_all_publication_figures(
            aggregated_results=aggregated,
            raw_results=dict(raw_results),
            save_dir=figures_dir,
        )
        print(f"Figures saved to: {figures_dir}")
    except Exception as e:
        print(f"Figure generation error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Done! Total time: {total_time/60:.1f} min")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
