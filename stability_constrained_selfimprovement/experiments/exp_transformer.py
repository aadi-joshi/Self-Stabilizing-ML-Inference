# ============================================================================
# Experiment B: Transformer on Algorithmic Tasks
# Sequential copy → reverse → sort with representation drift measurement
# ============================================================================

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.transformer import AlgorithmicTransformer, generate_algorithmic_data
from metrics.functional_drift import FunctionalDrift, RepresentationDrift
from metrics.constrained_optimizer import (
    StabilityConstrainedOptimizer, EpsilonScheduler, EWCRegularizer
)
from metrics.experiment_metrics import ExperimentMetrics
from utils.common import set_seed, AverageMeter, count_parameters


def run_transformer_experiment(
    method: str,
    config: Dict,
    seed: int,
    device: torch.device,
    save_dir: str,
) -> ExperimentMetrics:
    """
    Run transformer algorithmic task experiment.
    
    Sequential training on copy → reverse → sort.
    Measures how well the model retains previous task knowledge.
    """
    set_seed(seed)
    exp_cfg = config['experiment_b']
    model_cfg = exp_cfg['model']

    # Build transformer
    model = AlgorithmicTransformer(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        max_seq_len=model_cfg['max_seq_len'],
    ).to(device)
    print(f"  Transformer: {count_parameters(model):,} parameters")

    lr = exp_cfg.get('lr', 0.0005)
    if method == "weight_decay":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    tasks = exp_cfg.get('tasks', ['copy', 'reverse', 'sort'])
    epochs_per_task = exp_cfg.get('epochs_per_task', 50)
    batch_size = exp_cfg.get('batch_size', 64)
    seq_len = model_cfg.get('max_seq_len', 32)
    vocab_size = model_cfg.get('vocab_size', 16)

    # Generate fixed reference data for drift measurement
    ref_x, _ = generate_algorithmic_data('copy', 512, seq_len, vocab_size, device)

    # Method-specific setup
    drift_module = FunctionalDrift(
        reference_model=model, reference_data=ref_x,
        norm_type=config['drift'].get('norm_type', 'l2'), device=device
    )
    repr_drift = RepresentationDrift(
        reference_model=model, reference_data=ref_x, device=device
    )

    constrained_opt = None
    ewc_reg = None

    if method == "ewc":
        ewc_reg = EWCRegularizer(model, ewc_lambda=config['experiment_a'].get('ewc_lambda', 1000.0))

    metrics = ExperimentMetrics(seed=seed, method=method, experiment='transformer_algorithmic')
    initial_model = copy.deepcopy(model)
    best_task_acc: Dict[str, float] = {}
    global_step = 0

    for task_idx, task_name in enumerate(tasks):
        print(f"  Task {task_idx}: {task_name}")

        # Setup constraint for tasks after the first
        if method == "functional_trust" and task_idx > 0:
            drift_module = FunctionalDrift(
                reference_model=model, reference_data=ref_x,
                norm_type=config['drift'].get('norm_type', 'l2'), device=device
            )
            eps_scheduler = EpsilonScheduler(
                schedule_type=config['epsilon_scheduler'].get('type', 'fixed'),
                epsilon_init=exp_cfg.get('drift_epsilon', 0.3),
                epsilon_min=config['drift'].get('epsilon_min', 0.01),
                total_steps=epochs_per_task * 100,
            )
            constrained_opt = StabilityConstrainedOptimizer(
                model=model, base_optimizer=optimizer, drift_module=drift_module,
                lambda_init=exp_cfg.get('drift_lambda', 0.5),
                lambda_lr=config['drift'].get('lambda_lr', 0.01),
                epsilon_scheduler=eps_scheduler,
                grad_clip=config['training'].get('grad_clip', 1.0),
            )
        else:
            constrained_opt = None

        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()

            # Generate fresh data each epoch (algorithmic tasks = infinite data)
            n_batches = 100
            for _ in range(n_batches):
                global_step += 1
                x, y = generate_algorithmic_data(task_name, batch_size, seq_len, vocab_size, device)

                output = model(x)  # (B, S, V)
                task_loss = loss_fn(output.view(-1, vocab_size), y.view(-1))

                if method == "ewc" and ewc_reg is not None and task_idx > 0:
                    task_loss = task_loss + ewc_reg.penalty(model)

                if constrained_opt is not None:
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

                if global_step % config['metrics'].get('log_interval', 50) == 0:
                    drift_info = drift_module.compute(model)
                    repr_info = repr_drift.compute(model)
                    log_data = {
                        'task_loss': epoch_loss.avg,
                        'accuracy': acc,
                        'functional_drift': drift_info['drift'],
                        'cka_similarity': repr_info['cka_similarity'],
                        'param_norm_drift': sum(
                            (p1 - p2).norm().item()
                            for p1, p2 in zip(model.parameters(), initial_model.parameters())
                        ),
                    }
                    if constrained_opt is not None:
                        log_data['lambda'] = constrained_opt.lambda_val
                        log_data['epsilon'] = constrained_opt.epsilon
                    metrics.log_step(global_step, log_data)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: loss={epoch_loss.avg:.4f}, acc={epoch_acc.avg:.4f}")

        # Evaluate on all tasks
        for eval_task in tasks[:task_idx + 1]:
            eval_acc = evaluate_task(model, eval_task, seq_len, vocab_size, device, loss_fn)
            task_key = f"task_{eval_task}"
            if task_key not in metrics.task_accuracies:
                metrics.task_accuracies[task_key] = []
            metrics.task_accuracies[task_key].append(eval_acc)

            if eval_task not in best_task_acc or eval_acc > best_task_acc[eval_task]:
                best_task_acc[eval_task] = eval_acc

            if eval_task != task_name:
                metrics.compute_forgetting(task_key, best_task_acc[eval_task], eval_acc)

            print(f"    {eval_task} acc: {eval_acc:.4f}")

        # EWC Fisher estimation
        if method == "ewc" and ewc_reg is not None:
            # Create a temporary loader for Fisher estimation
            temp_x, temp_y = generate_algorithmic_data(task_name, 1000, seq_len, vocab_size, device)
            temp_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(temp_x, temp_y.view(-1, seq_len)[:, 0]),
                batch_size=128
            )
            # We need a custom Fisher estimation for seq2seq
            _estimate_fisher_seq2seq(ewc_reg, model, task_name, seq_len, vocab_size, device)

    metrics_path = os.path.join(save_dir, f"{method}_seed{seed}_metrics.json")
    metrics.save(metrics_path)
    return metrics


@torch.no_grad()
def evaluate_task(model, task_name, seq_len, vocab_size, device, loss_fn, n_eval=1000):
    """Evaluate model on a specific algorithmic task."""
    model.eval()
    x, y = generate_algorithmic_data(task_name, n_eval, seq_len, vocab_size, device)
    output = model(x)
    pred = output.argmax(dim=-1)
    # Token-level accuracy
    acc = (pred == y).float().mean().item()
    # Sequence-level accuracy (entire sequence correct)
    seq_acc = (pred == y).all(dim=1).float().mean().item()
    return acc


def _estimate_fisher_seq2seq(ewc_reg, model, task_name, seq_len, vocab_size, device, n_samples=500):
    """Estimate Fisher for sequence-to-sequence tasks."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    for _ in range(n_samples // 32 + 1):
        x, y = generate_algorithmic_data(task_name, 32, seq_len, vocab_size, device)
        model.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2) * 32

    for n in fisher:
        fisher[n] /= n_samples

    ewc_reg.fisher_diag = fisher
    ewc_reg.saved_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
