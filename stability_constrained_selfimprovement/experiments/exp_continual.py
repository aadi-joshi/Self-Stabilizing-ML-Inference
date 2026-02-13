# ============================================================================
# Experiment A: Continual Supervised Learning
# Sequential CIFAR-10 tasks with forgetting measurement
# ============================================================================

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.resnet import build_resnet
from metrics.functional_drift import FunctionalDrift, RepresentationDrift
from metrics.constrained_optimizer import (
    StabilityConstrainedOptimizer, EpsilonScheduler, EWCRegularizer
)
from metrics.experiment_metrics import ExperimentMetrics, StatisticalAnalyzer
from utils.common import set_seed, get_device, ensure_dir, count_parameters, AverageMeter


def get_cifar10_split_tasks(
    n_tasks: int = 5,
    data_dir: str = "./data",
    batch_size: int = 128,
) -> List[Dict]:
    """
    Split CIFAR-10 into n_tasks sequential binary classification tasks.
    Task 0: classes 0,1  |  Task 1: classes 2,3  |  ...  |  Task 4: classes 8,9
    
    Returns list of dicts, each with 'train_loader', 'test_loader', 'classes'.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    classes_per_task = 10 // n_tasks
    tasks = []

    for task_id in range(n_tasks):
        task_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

        # Filter indices
        train_indices = [i for i, (_, y) in enumerate(train_data) if y in task_classes]
        test_indices = [i for i, (_, y) in enumerate(test_data) if y in task_classes]

        # Remap labels to 0..classes_per_task-1
        class_map = {c: i for i, c in enumerate(task_classes)}

        # Create subsets with remapped labels
        train_x = torch.stack([train_data[i][0] for i in train_indices])
        train_y = torch.tensor([class_map[train_data[i][1]] for i in train_indices])
        test_x = torch.stack([test_data[i][0] for i in test_indices])
        test_y = torch.tensor([class_map[test_data[i][1]] for i in test_indices])

        train_loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False
        )

        tasks.append({
            'train_loader': train_loader,
            'test_loader': test_loader,
            'classes': task_classes,
            'task_id': task_id,
            'train_x': train_x,
            'test_x': test_x,
        })

    return tasks


def run_continual_learning(
    method: str,
    config: Dict,
    seed: int,
    device: torch.device,
    save_dir: str,
) -> ExperimentMetrics:
    """
    Run one complete continual learning experiment.
    
    Args:
        method: One of 'baseline', 'weight_decay', 'ewc', 'functional_trust'
        config: Experiment configuration
        seed: Random seed
        device: Torch device
        save_dir: Directory for saving results
    """
    set_seed(seed)
    exp_cfg = config['experiment_a']

    # Build model
    n_tasks = exp_cfg.get('num_tasks', 5)
    classes_per_task = 10 // n_tasks
    model = build_resnet(exp_cfg.get('model', 'resnet18_small'), num_classes=classes_per_task)
    model = model.to(device)
    print(f"  Model: {count_parameters(model):,} parameters")

    # Get tasks
    tasks = get_cifar10_split_tasks(n_tasks=n_tasks, batch_size=exp_cfg.get('batch_size', 128))

    # Setup optimizer
    lr = exp_cfg.get('lr', 0.001)
    if method == "weight_decay":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    # Method-specific setup
    drift_module = None
    constrained_opt = None
    ewc_reg = None
    repr_drift = None

    if method == "ewc":
        ewc_reg = EWCRegularizer(model, ewc_lambda=exp_cfg.get('ewc_lambda', 1000.0))

    # Metrics
    metrics = ExperimentMetrics(seed=seed, method=method, experiment='continual_cifar')
    initial_model = copy.deepcopy(model)

    # Track best accuracy per task (for forgetting computation)
    best_task_acc: Dict[int, float] = {}
    global_step = 0

    # --- Sequential Task Training ---
    for task_id, task in enumerate(tasks):
        print(f"  Task {task_id}/{n_tasks - 1} (classes {task['classes']})")

        # Setup drift constraint at start of each task (except first for some methods)
        if method == "functional_trust" and task_id > 0:
            # Get reference data from current task
            ref_data = task['train_x'][:config['drift']['num_reference_points']].to(device)

            drift_module = FunctionalDrift(
                reference_model=model, reference_data=ref_data,
                norm_type=config['drift'].get('norm_type', 'l2'), device=device
            )
            repr_drift = RepresentationDrift(
                reference_model=model, reference_data=ref_data, device=device
            )

            eps_scheduler = EpsilonScheduler(
                schedule_type=config['epsilon_scheduler'].get('type', 'fixed'),
                epsilon_init=exp_cfg.get('drift_epsilon', 0.5),
                epsilon_min=config['drift'].get('epsilon_min', 0.01),
                epsilon_max=config['drift'].get('epsilon_max', 10.0),
                warmup_steps=config['epsilon_scheduler'].get('warmup_steps', 100),
                total_steps=len(task['train_loader']) * exp_cfg.get('epochs_per_task', 30),
            )

            constrained_opt = StabilityConstrainedOptimizer(
                model=model, base_optimizer=optimizer, drift_module=drift_module,
                lambda_init=exp_cfg.get('drift_lambda', 1.0),
                lambda_lr=config['drift'].get('lambda_lr', 0.01),
                epsilon_scheduler=eps_scheduler,
                grad_clip=config['training'].get('grad_clip', 1.0),
            )
        elif method == "functional_trust" and task_id == 0:
            # First task: no constraint (nothing to preserve)
            ref_data = task['train_x'][:config['drift']['num_reference_points']].to(device)
            drift_module = FunctionalDrift(
                reference_model=model, reference_data=ref_data,
                norm_type=config['drift'].get('norm_type', 'l2'), device=device
            )
            repr_drift = RepresentationDrift(
                reference_model=model, reference_data=ref_data, device=device
            )
            constrained_opt = None

        # Train on current task
        epochs = exp_cfg.get('epochs_per_task', 30)
        for epoch in range(epochs):
            model.train()
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()

            for x, y in task['train_loader']:
                global_step += 1
                x, y = x.to(device), y.to(device)

                output = model(x)
                task_loss = loss_fn(output, y)

                # Apply method-specific regularization
                if method == "ewc" and ewc_reg is not None:
                    task_loss = task_loss + ewc_reg.penalty(model)

                if method == "functional_trust" and constrained_opt is not None:
                    step_metrics = constrained_opt.step(task_loss)
                else:
                    optimizer.zero_grad()
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    step_metrics = {'task_loss': task_loss.item()}

                pred = output.argmax(dim=-1)
                acc = (pred == y).float().mean().item()
                epoch_loss.update(step_metrics.get('task_loss', task_loss.item()))
                epoch_acc.update(acc)

                # Log periodically
                if global_step % config['metrics'].get('log_interval', 50) == 0:
                    log_data = {
                        'task_loss': epoch_loss.avg,
                        'accuracy': acc,
                        'grad_norm': sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None),
                        'param_norm_drift': sum(
                            (p1 - p2).norm().item()
                            for p1, p2 in zip(model.parameters(), initial_model.parameters())
                        ),
                    }
                    if drift_module is not None:
                        drift_info = drift_module.compute(model)
                        log_data['functional_drift'] = drift_info['drift']
                    if constrained_opt is not None:
                        log_data['lambda'] = constrained_opt.lambda_val
                        log_data['epsilon'] = constrained_opt.epsilon
                    if repr_drift is not None:
                        repr_info = repr_drift.compute(model)
                        log_data['cka_similarity'] = repr_info['cka_similarity']

                    metrics.log_step(global_step, log_data)

        # After task training: evaluate on ALL tasks seen so far
        for prev_id in range(task_id + 1):
            eval_result = evaluate_model(model, tasks[prev_id]['test_loader'], loss_fn, device)
            task_key = f"task_{prev_id}"
            if task_key not in metrics.task_accuracies:
                metrics.task_accuracies[task_key] = []
            metrics.task_accuracies[task_key].append(eval_result['accuracy'])

            # Track best accuracy
            if prev_id not in best_task_acc or eval_result['accuracy'] > best_task_acc[prev_id]:
                best_task_acc[prev_id] = eval_result['accuracy']

            # Compute forgetting for previous tasks
            if prev_id < task_id:
                forgetting = best_task_acc[prev_id] - eval_result['accuracy']
                metrics.compute_forgetting(task_key, best_task_acc[prev_id], eval_result['accuracy'])

            print(f"    Task {prev_id} acc: {eval_result['accuracy']:.4f}")

        # EWC: estimate Fisher after each task
        if method == "ewc" and ewc_reg is not None:
            ewc_reg.estimate_fisher(model, task['train_loader'], device)

    # Save metrics
    metrics_path = os.path.join(save_dir, f"{method}_seed{seed}_metrics.json")
    metrics.save(metrics_path)
    print(f"  Saved metrics to {metrics_path}")

    return metrics


@torch.no_grad()
def evaluate_model(model, data_loader, loss_fn, device) -> Dict[str, float]:
    """Evaluate model accuracy and loss."""
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        pred = output.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        total_loss += loss.item() * x.shape[0]
        total_samples += x.shape[0]

    return {
        'accuracy': total_correct / max(total_samples, 1),
        'loss': total_loss / max(total_samples, 1),
    }
