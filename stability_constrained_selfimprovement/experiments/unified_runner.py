# ============================================================================
# Unified Continual Learning Experiment Runner
#
# Supports all benchmarks × all methods with standardized evaluation.
#
# Benchmarks: split_cifar10, split_cifar100, permuted_mnist, rotated_mnist
# Methods: baseline, weight_decay, ewc, si, lwf, distillation, replay,
#          functional_trust, feature_trust
#
# Evaluation: Average accuracy, backward transfer, forward transfer,
#             forgetting measure, stability-plasticity curves
# ============================================================================

import os
import sys
import copy
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.resnet import build_resnet
from metrics.functional_drift import FunctionalDrift, FeatureFunctionalDrift, OnlineDistillationDrift, RepresentationDrift
from metrics.constrained_optimizer import (
    StabilityConstrainedOptimizer, EpsilonScheduler, EWCRegularizer
)
from metrics.baselines import (
    SynapticIntelligence, LearningWithoutForgetting,
    FixedDistillation, ExperienceReplay, FeatureSpaceDrift
)
from metrics.experiment_metrics import ExperimentMetrics
from experiments.benchmarks import (
    get_permuted_mnist_tasks, get_rotated_mnist_tasks,
    get_split_cifar100_tasks, MNISTResNet
)
from experiments.exp_continual import get_cifar10_split_tasks
from utils.common import set_seed, get_device, ensure_dir, count_parameters, AverageMeter


# All supported methods
ALL_METHODS = [
    'baseline', 'weight_decay', 'ewc', 'si', 'lwf',
    'distillation', 'replay', 'functional_trust', 'feature_trust',
    'ftr_replay',  # FTR + replay hybrid
]

# All supported benchmarks  
ALL_BENCHMARKS = [
    'split_cifar10', 'split_cifar100', 'permuted_mnist', 'rotated_mnist',
]


def get_benchmark_tasks(benchmark: str, config: Dict) -> Tuple[List[Dict], nn.Module]:
    """
    Get tasks and model for a benchmark.
    
    Returns:
        tasks: List of task dicts
        model: Appropriate model architecture
    """
    if benchmark == 'split_cifar10':
        n_tasks = config.get('experiment_a', {}).get('num_tasks', 5)
        batch_size = config.get('experiment_a', {}).get('batch_size', 128)
        tasks = get_cifar10_split_tasks(n_tasks=n_tasks, batch_size=batch_size)
        classes_per_task = 10 // n_tasks
        model_variant = config.get('experiment_a', {}).get('model', 'resnet18_small')
        model = build_resnet(model_variant, num_classes=classes_per_task)
        return tasks, model

    elif benchmark == 'split_cifar100':
        n_tasks = config.get('cifar100', {}).get('num_tasks', 10)
        batch_size = config.get('experiment_a', {}).get('batch_size', 128)
        tasks = get_split_cifar100_tasks(n_tasks=n_tasks, batch_size=batch_size)
        classes_per_task = 100 // n_tasks
        model_variant = config.get('experiment_a', {}).get('model', 'resnet18_small')
        model = build_resnet(model_variant, num_classes=classes_per_task)
        return tasks, model

    elif benchmark == 'permuted_mnist':
        n_tasks = config.get('mnist', {}).get('num_tasks', 10)
        batch_size = config.get('mnist', {}).get('batch_size', 128)
        seed = config.get('seed', 42)
        tasks = get_permuted_mnist_tasks(n_tasks=n_tasks, batch_size=batch_size, seed=seed)
        model = MNISTResNet(num_classes=10, base_width=16)
        return tasks, model

    elif benchmark == 'rotated_mnist':
        batch_size = config.get('mnist', {}).get('batch_size', 128)
        tasks = get_rotated_mnist_tasks(batch_size=batch_size)
        model = MNISTResNet(num_classes=10, base_width=16)
        return tasks, model

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def run_unified_experiment(
    benchmark: str,
    method: str,
    config: Dict,
    seed: int,
    device: torch.device,
    save_dir: str,
    verbose: bool = True,
) -> Dict:
    """
    Run a single experiment: one benchmark × one method × one seed.
    
    Returns a comprehensive results dict with:
        - accuracy_matrix: (n_tasks, n_tasks) matrix of acc[trained_on][evaluated_on]
        - average_accuracy: mean final accuracy across all tasks
        - backward_transfer: BWT metric
        - forward_transfer: FWT metric
        - forgetting: average forgetting measure
        - per_step_metrics: time series of training metrics
        - lambda_history: (FTR only) dual variable trajectory
        - drift_history: functional drift time series
        - cka_history: CKA similarity time series
    """
    set_seed(seed)
    
    # Get tasks and model
    tasks, model = get_benchmark_tasks(benchmark, config)
    model = model.to(device)
    n_tasks = len(tasks)
    
    if verbose:
        print(f"  Benchmark: {benchmark} ({n_tasks} tasks)")
        print(f"  Method: {method}")
        print(f"  Model: {count_parameters(model):,} parameters")
        print(f"  Device: {device}")

    # Get config params
    exp_cfg = config.get('experiment_a', {})
    lr = exp_cfg.get('lr', 0.001)
    epochs_per_task = exp_cfg.get('epochs_per_task', 30)
    n_ref_points = config.get('drift', {}).get('num_reference_points', 512)

    # Setup optimizer
    if method == "weight_decay":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    # Initialize method-specific components
    ewc_reg = None
    si_reg = None
    lwf_reg = None
    distill_reg = None
    replay_buffer = None
    drift_module = None
    feature_drift_module = None
    constrained_opt = None
    repr_drift = None

    if method == 'ewc':
        ewc_reg = EWCRegularizer(model, ewc_lambda=exp_cfg.get('ewc_lambda', 1000.0))
    elif method == 'si':
        si_reg = SynapticIntelligence(model, si_c=config.get('si', {}).get('c', 1.0))
    elif method == 'lwf':
        lwf_reg = LearningWithoutForgetting(
            model, lwf_alpha=config.get('lwf', {}).get('alpha', 1.0),
            temperature=config.get('lwf', {}).get('temperature', 2.0),
        )
    elif method == 'distillation':
        distill_reg = FixedDistillation(
            model, distill_lambda=config.get('distillation', {}).get('lambda', 1.0),
        )
    elif method == 'replay':
        replay_buffer = ExperienceReplay(
            buffer_size=config.get('replay', {}).get('buffer_size', 500),
            replay_batch_size=config.get('replay', {}).get('batch_size', 32),
        )
    elif method == 'ftr_replay':
        # FTR + Experience Replay hybrid: both functional constraint AND replay
        replay_buffer = ExperienceReplay(
            buffer_size=config.get('replay', {}).get('buffer_size', 500),
            replay_batch_size=config.get('replay', {}).get('batch_size', 32),
        )

    # Tracking
    accuracy_matrix = np.zeros((n_tasks, n_tasks))  # [trained_up_to][evaluated_on]
    best_task_acc = {}
    per_step_metrics = []
    lambda_history = []
    drift_history = []
    cka_history = []
    initial_model = copy.deepcopy(model)
    global_step = 0
    task_boundaries = []
    
    # Cumulative reference data: samples from ALL previous tasks
    # This is critical — we need to constrain drift on old task distributions
    cumulative_ref_x = []

    # ========= Sequential Task Training =========
    for task_id in range(n_tasks):
        task = tasks[task_id]
        task_start_time = time.time()
        
        if verbose:
            print(f"\n  --- Task {task_id}/{n_tasks-1} ---")

        # Record task boundary
        task_boundaries.append(global_step)

        # Build reference data from ALL previous tasks (balanced sampling)
        if task_id > 0 and method in ('functional_trust', 'feature_trust', 'ftr_replay'):
            # Use cumulative reference data from previous tasks
            ref_per_task = max(50, n_ref_points // task_id)
            ref_parts = []
            for prev_id in range(task_id):
                prev_x = tasks[prev_id]['train_x']
                n_avail = min(ref_per_task, prev_x.shape[0])
                ref_parts.append(prev_x[:n_avail])
            ref_data = torch.cat(ref_parts, dim=0).to(device)
        else:
            ref_data = task['train_x'][:n_ref_points].to(device)
        
        if method in ('functional_trust', 'feature_trust', 'ftr_replay') and task_id > 0:
            if method in ('functional_trust', 'ftr_replay'):
                # Use ONLINE DISTILLATION drift — KL divergence on current batch
                # This is the core FTR design: same distillation signal as LwF
                # but with adaptive Lagrangian multiplier λ instead of fixed α.
                # The Lagrangian framework automatically tunes the trade-off,
                # providing principled stability-plasticity balancing.
                drift_module = OnlineDistillationDrift(
                    reference_model=model, reference_data=ref_data,
                    norm_type='kl', device=device,
                    temperature=config.get('lwf', {}).get('temperature', 2.0),
                )
            else:  # feature_trust
                feature_drift_module = FeatureSpaceDrift(
                    model=model, reference_data=ref_data, device=device,
                )
                # Wrap as drift_module interface
                drift_module = FunctionalDrift(
                    reference_model=model, reference_data=ref_data,
                    norm_type='l2', device=device,
                )

            repr_drift = RepresentationDrift(
                reference_model=model, reference_data=ref_data, device=device,
            )

            eps_scheduler = EpsilonScheduler(
                schedule_type=config.get('epsilon_scheduler', {}).get('type', 'fixed'),
                epsilon_init=exp_cfg.get('drift_epsilon', 0.5),
                epsilon_min=config.get('drift', {}).get('epsilon_min', 0.01),
                epsilon_max=config.get('drift', {}).get('epsilon_max', 10.0),
                warmup_steps=config.get('epsilon_scheduler', {}).get('warmup_steps', 0),
                total_steps=len(task['train_loader']) * epochs_per_task,
            )

            if method in ('functional_trust', 'ftr_replay'):
                constrained_opt = StabilityConstrainedOptimizer(
                    model=model, base_optimizer=optimizer,
                    drift_module=drift_module,
                    lambda_init=exp_cfg.get('drift_lambda', 1.0),
                    lambda_lr=config.get('drift', {}).get('lambda_lr', 0.01),
                    lambda_momentum=config.get('drift', {}).get('lambda_momentum', 0.9),
                    epsilon_scheduler=eps_scheduler,
                    grad_clip=config.get('training', {}).get('grad_clip', 1.0),
                )
            else:
                constrained_opt = None
        else:
            drift_module = FunctionalDrift(
                reference_model=model, reference_data=ref_data,
                norm_type='l2', device=device,
            ) if task_id == 0 else drift_module
            repr_drift = RepresentationDrift(
                reference_model=model, reference_data=ref_data, device=device,
            ) if task_id == 0 else repr_drift
            constrained_opt = None

        # LwF / Distillation: save old model before new task
        if method == 'lwf' and lwf_reg is not None and task_id > 0:
            lwf_reg.begin_new_task(model)
        if method == 'distillation' and distill_reg is not None and task_id > 0:
            distill_reg.begin_new_task(model)

        # ========= Train on current task =========
        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()

            for x, y in task['train_loader']:
                global_step += 1
                x, y = x.to(device), y.to(device)

                output = model(x)
                task_loss = loss_fn(output, y)

                # === Method-specific modifications ===
                total_loss = task_loss

                if method == 'ewc' and ewc_reg is not None and task_id > 0:
                    total_loss = total_loss + ewc_reg.penalty(model)

                elif method == 'si' and si_reg is not None and task_id > 0:
                    total_loss = total_loss + si_reg.penalty(model)

                elif method == 'lwf' and lwf_reg is not None and task_id > 0:
                    total_loss = total_loss + lwf_reg.distillation_loss(model, x)

                elif method == 'distillation' and distill_reg is not None and task_id > 0:
                    total_loss = total_loss + distill_reg.penalty(model, x)

                elif method == 'replay' and replay_buffer is not None and task_id > 0:
                    total_loss = total_loss + replay_buffer.replay_loss(model, loss_fn, device)

                elif method == 'feature_trust' and feature_drift_module is not None and task_id > 0:
                    feat_drift = feature_drift_module.compute_differentiable(model)
                    total_loss = total_loss + exp_cfg.get('drift_lambda', 1.0) * feat_drift

                # FTR+Replay: add replay loss, then use constrained optimizer for drift
                elif method == 'ftr_replay' and replay_buffer is not None and task_id > 0:
                    replay_loss_val = replay_buffer.replay_loss(model, loss_fn, device)
                    task_loss = task_loss + replay_loss_val  # Augment task loss with replay

                # FTR and FTR+Replay use their own step logic via constrained optimizer
                if method in ('functional_trust', 'ftr_replay') and constrained_opt is not None:
                    step_info = constrained_opt.step(task_loss, current_batch=x)
                    lambda_history.append(step_info.get('lambda', 0))
                else:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # SI: track importance online
                if method == 'si' and si_reg is not None:
                    si_reg.update_running_importance(model)

                # Track metrics
                pred = output.argmax(dim=-1)
                acc = (pred == y).float().mean().item()
                epoch_loss.update(task_loss.item())
                epoch_acc.update(acc)

                # Periodic logging
                log_interval = config.get('metrics', {}).get('log_interval', 50)
                if global_step % log_interval == 0:
                    step_data = {
                        'step': global_step,
                        'task_id': task_id,
                        'task_loss': epoch_loss.avg,
                        'accuracy': acc,
                    }

                    if drift_module is not None:
                        try:
                            drift_info = drift_module.compute(model)
                            step_data['functional_drift'] = drift_info['drift']
                            drift_history.append(drift_info['drift'])
                        except Exception:
                            pass

                    if repr_drift is not None:
                        try:
                            repr_info = repr_drift.compute(model)
                            step_data['cka_similarity'] = repr_info['cka_similarity']
                            cka_history.append(repr_info['cka_similarity'])
                        except Exception:
                            pass

                    if constrained_opt is not None:
                        step_data['lambda'] = constrained_opt.lambda_val
                        step_data['epsilon'] = constrained_opt.epsilon

                    per_step_metrics.append(step_data)

            # End of epoch logging
            if verbose and (epoch % max(1, epochs_per_task // 3) == 0 or epoch == epochs_per_task - 1):
                print(f"    Epoch {epoch}: loss={epoch_loss.avg:.4f}, acc={epoch_acc.avg:.4f}")

        # ========= Post-task operations =========
        
        # EWC: estimate Fisher
        if method == 'ewc' and ewc_reg is not None:
            ewc_reg.estimate_fisher(model, task['train_loader'], device)

        # SI: consolidate importance
        if method == 'si' and si_reg is not None:
            si_reg.consolidate(model)

        # Replay / FTR+Replay: add task data to buffer
        if method in ('replay', 'ftr_replay') and replay_buffer is not None:
            replay_buffer.add_task_data(
                task['train_x'][:1000],
                torch.tensor([task['train_loader'].dataset[i][1] for i in range(min(1000, len(task['train_loader'].dataset)))]),
                task_budget=500 // (task_id + 1),
            )

        # ========= Evaluate on ALL tasks seen so far =========
        for eval_id in range(task_id + 1):
            eval_acc = _evaluate(model, tasks[eval_id]['test_loader'], device)
            accuracy_matrix[task_id, eval_id] = eval_acc

            if eval_id not in best_task_acc or eval_acc > best_task_acc[eval_id]:
                best_task_acc[eval_id] = eval_acc

            if verbose:
                print(f"    Task {eval_id} acc: {eval_acc:.4f}", end="")
                if eval_id < task_id and eval_id in best_task_acc:
                    fgt = best_task_acc[eval_id] - eval_acc
                    print(f" (forgetting: {fgt:.4f})", end="")
                print()

        task_time = time.time() - task_start_time
        if verbose:
            print(f"    Task time: {task_time:.1f}s")

    # ========= Compute aggregate metrics =========
    results = _compute_aggregate_metrics(accuracy_matrix, n_tasks)
    results.update({
        'benchmark': benchmark,
        'method': method,
        'seed': seed,
        'accuracy_matrix': accuracy_matrix.tolist(),
        'per_step_metrics': per_step_metrics,
        'lambda_history': lambda_history,
        'drift_history': drift_history,
        'cka_history': cka_history,
        'task_boundaries': task_boundaries,
        'n_params': count_parameters(model),
    })

    # Save
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, f"{benchmark}_{method}_seed{seed}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n  === Final Metrics ===")
        print(f"  Average Accuracy: {results['average_accuracy']:.4f}")
        print(f"  Backward Transfer: {results['backward_transfer']:.4f}")
        print(f"  Forward Transfer: {results['forward_transfer']:.4f}")
        print(f"  Forgetting: {results['forgetting']:.4f}")

    return results


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        pred = output.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    model.train()
    return correct / max(total, 1)


def _compute_aggregate_metrics(acc_matrix: np.ndarray, n_tasks: int) -> Dict:
    """
    Compute standard continual learning metrics from the accuracy matrix.
    
    acc_matrix[i, j] = accuracy on task j after training on tasks 0..i
    
    Metrics:
        Average Accuracy (AA): mean of final row
        Backward Transfer (BWT): mean of (final_acc - best_acc) for old tasks
        Forward Transfer (FWT): mean of first-time accuracy minus random
        Forgetting (F): mean of (best_acc - final_acc) for old tasks
    """
    # Average accuracy: mean of the last row
    average_accuracy = acc_matrix[n_tasks - 1, :].mean()

    # Backward transfer
    bwt_values = []
    for j in range(n_tasks - 1):
        # Best accuracy on task j (at any point after learning it)
        best_j = max(acc_matrix[i, j] for i in range(j, n_tasks))
        final_j = acc_matrix[n_tasks - 1, j]
        bwt_values.append(final_j - best_j)
    backward_transfer = np.mean(bwt_values) if bwt_values else 0.0

    # Forward transfer: accuracy on task j when first encountering it
    # (before any training on it) vs random baseline
    fwt_values = []
    for j in range(1, n_tasks):
        # Zero-shot accuracy on task j (from training on tasks 0..j-1)
        zero_shot = acc_matrix[j - 1, j] if j < acc_matrix.shape[1] else 0
        # Random baseline depends on number of classes
        fwt_values.append(zero_shot)
    forward_transfer = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting
    forgetting_values = []
    for j in range(n_tasks - 1):
        best_j = max(acc_matrix[i, j] for i in range(j, n_tasks))
        final_j = acc_matrix[n_tasks - 1, j]
        forgetting_values.append(max(0, best_j - final_j))
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'average_accuracy': float(average_accuracy),
        'backward_transfer': float(backward_transfer),
        'forward_transfer': float(forward_transfer),
        'forgetting': float(forgetting),
    }


def run_full_benchmark_suite(
    benchmarks: List[str],
    methods: List[str],
    config: Dict,
    seeds: List[int],
    save_dir: str,
    device: torch.device,
    verbose: bool = True,
) -> Dict:
    """
    Run the full Cartesian product: benchmarks × methods × seeds.
    
    Returns aggregated results across seeds.
    """
    all_results = defaultdict(lambda: defaultdict(list))
    
    total_runs = len(benchmarks) * len(methods) * len(seeds)
    completed = 0
    
    for benchmark in benchmarks:
        for method in methods:
            for seed in seeds:
                completed += 1
                print(f"\n{'='*70}")
                print(f"[{completed}/{total_runs}] {benchmark} | {method} | seed={seed}")
                print(f"{'='*70}")
                
                try:
                    result = run_unified_experiment(
                        benchmark=benchmark,
                        method=method,
                        config=config,
                        seed=seed,
                        device=device,
                        save_dir=os.path.join(save_dir, benchmark),
                        verbose=verbose,
                    )
                    all_results[benchmark][method].append(result)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Aggregate across seeds
    aggregated = {}
    for benchmark in benchmarks:
        aggregated[benchmark] = {}
        for method in methods:
            results_list = all_results[benchmark][method]
            if results_list:
                aggregated[benchmark][method] = _aggregate_results(results_list)

    # Save aggregated results
    ensure_dir(save_dir)
    agg_path = os.path.join(save_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def _aggregate_results(results_list: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
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
        agg['accuracy_matrix_std'] = stacked.std(axis=0, ddof=1).tolist() if len(matrices) > 1 else np.zeros_like(matrices[0]).tolist()

    return agg
