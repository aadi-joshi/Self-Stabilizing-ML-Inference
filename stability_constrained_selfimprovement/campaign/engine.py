"""
Unified continual-learning training engine.

Ported from stability_constrained_selfimprovement/run_neurips_breakthrough.py
(run_cl_experiment) with the following extensions used by the campaign:

  1. `kl_direction` in {'forward','reverse','js'} for FTR/LwF -- ablates the
     direction of the KL constraint (NEXT.md Sec 6.2). 'forward' matches the
     original paper: D_KL(f_old || f_theta) via
     F.kl_div(log f_theta, f_old.detach()).
  2. Full accuracy matrix + per-task forgetting profile + new-task accuracy
     (accuracy on task t measured immediately after training on task t,
     before any subsequent forgetting) are returned, not just the two
     aggregate scalars (NEXT.md Sec 6.1, 5.-aggregate-only limitation).
  3. Optional trajectory logging of (lambda_t, D_KL_t) per step, used by the
     optimizer-schedule diagnostic to determine whether ~* reflects task
     geometry or dual-ascent dynamics.
  4. `device` is a first-class argument (no hardcoded CPU).
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def _kl_term(new_logits, old_logits, T, direction):
    """Returns the (unreduced-then-batchmean) T^2-scaled KL/JS divergence term."""
    if direction == 'forward':
        # D_KL(old || new): matches original paper (kl_div(log_new, old.detach()))
        old_soft = F.softmax(old_logits / T, dim=-1)
        new_log = F.log_softmax(new_logits / T, dim=-1)
        return T * T * F.kl_div(new_log, old_soft, reduction='batchmean')
    elif direction == 'reverse':
        # D_KL(new || old): mode-seeking direction, opposite of the paper's default
        new_soft = F.softmax(new_logits / T, dim=-1)
        old_log = F.log_softmax(old_logits / T, dim=-1)
        return T * T * F.kl_div(old_log, new_soft, reduction='batchmean')
    elif direction == 'js':
        old_soft = F.softmax(old_logits / T, dim=-1)
        new_soft = F.softmax(new_logits / T, dim=-1)
        m = 0.5 * (old_soft + new_soft)
        log_m = (m.clamp_min(1e-12)).log()
        kl_old_m = F.kl_div(log_m, old_soft, reduction='batchmean')
        kl_new_m = F.kl_div(log_m, new_soft, reduction='batchmean')
        return T * T * 0.5 * (kl_old_m + kl_new_m)
    else:
        raise ValueError(direction)


def run_cl_experiment(tasks, model_factory, method, seed, device,
                       epochs_per_task=4, method_cfg=None, log_trajectory=False,
                       kl_direction='forward'):
    """
    Unified CL training supporting: ftr, ewc, lwf, si, replay, ftr_replay, finetune.

    Returns a dict with:
      average_accuracy, forgetting            (aggregate, as before)
      new_task_accuracy                       (mean of acc_matrix diagonal)
      per_task_forgetting                     (list, len n_tasks-1)
      acc_matrix                              (n_tasks x n_tasks list-of-lists)
      trajectory (optional)                   {'lambda': [...], 'drift': [...], 'step_task': [...]}
    """
    set_seed(seed)
    if method_cfg is None:
        method_cfg = {}

    n_tasks = len(tasks)
    nc = tasks[0]['num_classes']
    model = model_factory(nc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=method_cfg.get('lr', 0.001))
    loss_fn = nn.CrossEntropyLoss()

    old_model = None
    replay_buffer_x, replay_buffer_y = [], []
    ewc_fisher, ewc_params = {}, {}
    si_omega, si_old_params, si_w = {}, {}, {}
    acc_matrix = np.zeros((n_tasks, n_tasks))

    eps = method_cfg.get('epsilon', 0.2)
    lam_init = method_cfg.get('lambda_init', 1.0)
    lam_lr = method_cfg.get('lambda_lr', 0.005)
    lam_max = method_cfg.get('lambda_max', 50.0)
    momentum = method_cfg.get('lambda_momentum', 0.9)
    temp = method_cfg.get('temperature', 2.0)
    warmup_ep = method_cfg.get('warmup_epochs', 1)
    replay_size = method_cfg.get('replay_size', 500)

    trajectory = {'lambda': [], 'drift': [], 'step_task': []} if log_trajectory else None

    for task_id in range(n_tasks):
        task = tasks[task_id]

        if task_id > 0 and method in ('lwf', 'ftr', 'ftr_replay'):
            old_model = copy.deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        if method == 'si' and task_id > 0:
            si_old_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
            si_w = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        if task_id > 0 and method in ('ftr', 'ftr_replay'):
            lam = lam_init
            ema_viol = 0.0
            step_count = 0
            wb = warmup_ep * len(task['train_loader'])

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in task['train_loader']:
                x, y = x.to(device), y.to(device)
                output = model(x)
                task_loss = loss_fn(output, y)
                reg_loss = torch.tensor(0.0, device=device)

                if method == 'ewc' and task_id > 0 and ewc_fisher:
                    for n, p in model.named_parameters():
                        if n in ewc_fisher:
                            reg_loss = reg_loss + (ewc_fisher[n] * (p - ewc_params[n]).pow(2)).sum()
                    reg_loss = method_cfg.get('ewc_lambda', 400.0) * reg_loss

                elif method == 'si' and task_id > 0 and si_omega:
                    for n, p in model.named_parameters():
                        if n in si_omega:
                            reg_loss = reg_loss + (si_omega[n] * (p - si_old_params.get(n, p)).pow(2)).sum()
                    reg_loss = method_cfg.get('si_c', 0.5) * reg_loss

                elif method == 'lwf' and task_id > 0 and old_model is not None:
                    with torch.no_grad():
                        old_out = old_model(x)
                    alpha = method_cfg.get('lwf_alpha', 1.0)
                    reg_loss = alpha * _kl_term(output, old_out, temp, kl_direction)

                elif method == 'replay' and task_id > 0 and replay_buffer_x:
                    rbx = torch.cat(replay_buffer_x, 0)
                    rby = torch.cat(replay_buffer_y, 0)
                    idx = torch.randperm(rbx.shape[0])[:min(64, rbx.shape[0])]
                    reg_loss = loss_fn(model(rbx[idx].to(device)), rby[idx].to(device))

                if method in ('ftr', 'ftr_replay') and task_id > 0:
                    step_count += 1
                    with torch.no_grad():
                        old_out = old_model(x)
                    dv = _kl_term(output, old_out, temp, kl_direction)

                    rep_loss = torch.tensor(0.0, device=device)
                    if method == 'ftr_replay' and replay_buffer_x:
                        rbx = torch.cat(replay_buffer_x, 0)
                        rby = torch.cat(replay_buffer_y, 0)
                        idx = torch.randperm(rbx.shape[0])[:min(64, rbx.shape[0])]
                        rep_loss = loss_fn(model(rbx[idx].to(device)), rby[idx].to(device))

                    if step_count > wb:
                        total_loss = task_loss + lam * dv + rep_loss
                        viol = dv.item() - eps
                        ema_viol = momentum * ema_viol + (1 - momentum) * viol
                        lam = max(0.0, min(lam_max, lam + lam_lr * ema_viol))
                    else:
                        total_loss = task_loss + dv + rep_loss

                    if trajectory is not None:
                        trajectory['lambda'].append(float(lam))
                        trajectory['drift'].append(float(dv.item()))
                        trajectory['step_task'].append(task_id)
                else:
                    total_loss = task_loss + reg_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if method == 'si' and task_id > 0 and si_old_params:
                    for n, p in model.named_parameters():
                        if n in si_w and p.grad is not None:
                            si_w[n] += (-p.grad * (p - si_old_params.get(n, p))).detach()

        if method == 'ewc':
            fisher = {}
            model.eval()
            for x, y in task['train_loader']:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] = fisher.get(n, torch.zeros_like(p)) + p.grad.data.clone().pow(2)
            ns = len(task['train_loader'].dataset)
            for n in fisher:
                fisher[n] /= ns
            if ewc_fisher:
                for n in fisher:
                    ewc_fisher[n] = 0.5 * ewc_fisher.get(n, torch.zeros_like(fisher[n])) + 0.5 * fisher[n]
            else:
                ewc_fisher = fisher
            ewc_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

        if method == 'si' and si_w:
            xi = 1e-3
            for n, p in model.named_parameters():
                if n in si_w and n in si_old_params:
                    delta = (p - si_old_params[n]).pow(2) + xi
                    new_omega = si_w[n] / delta
                    si_omega[n] = si_omega.get(n, torch.zeros_like(p)) + new_omega.detach()
            si_old_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

        if method in ('ftr_replay', 'replay'):
            per_task = replay_size // (task_id + 1)
            n_store = min(per_task, len(task['train_loader'].dataset))
            replay_buffer_x = replay_buffer_x[:task_id]
            replay_buffer_y = replay_buffer_y[:task_id]
            replay_buffer_x.append(task['train_x'][:n_store].cpu())
            replay_buffer_y.append(task['train_y'][:n_store].cpu())

        model.eval()
        for eid in range(task_id + 1):
            acc_matrix[task_id, eid] = evaluate(model, tasks[eid]['test_loader'], device)

    result = compute_metrics(acc_matrix, n_tasks)
    result['acc_matrix'] = acc_matrix.tolist()
    if trajectory is not None:
        result['trajectory'] = trajectory
    return result


def compute_metrics(acc_matrix, n_tasks):
    aa = acc_matrix[n_tasks - 1, :n_tasks].mean()
    fgt_v = []
    for j in range(n_tasks - 1):
        best_j = max(acc_matrix[i, j] for i in range(j, n_tasks))
        fgt_v.append(max(0, best_j - acc_matrix[n_tasks - 1, j]))
    new_task_acc = float(np.mean([acc_matrix[t, t] for t in range(n_tasks)]))
    return {
        'average_accuracy': float(aa),
        'forgetting': float(np.mean(fgt_v)) if fgt_v else 0.0,
        'new_task_accuracy': new_task_acc,
        'per_task_forgetting': [float(v) for v in fgt_v],
    }


# ======================================================================
# Curvature measurement (ported from run_neurips_breakthrough.py)
# ======================================================================
from contextlib import contextmanager


@contextmanager
def _force_math_sdpa():
    """ViT's nn.MultiheadAttention dispatches to a fused scaled-dot-product-
    attention kernel whose backward is not itself twice-differentiable
    (needed for the Hutchinson/power-iteration Hessian estimators below,
    which call autograd.grad with create_graph=True). Force the plain
    math-mode SDPA backend for the duration of curvature measurement only;
    regular training is unaffected and keeps using the fused kernels."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):
            yield
        return
    except Exception:
        pass
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield
        return
    except Exception:
        yield


def compute_hessian_trace(model, loader, device, loss_fn, n_samples=10, max_batches=3):
    model.train()
    traces = []
    params = [p for p in model.parameters() if p.requires_grad]
    with _force_math_sdpa():
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            for _ in range(n_samples):
                model.zero_grad()
                loss = loss_fn(model(x), y)
                grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
                v = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in params]
                gv = sum((g * vi).sum() for g, vi in zip(grads, v) if g is not None)
                hvp = torch.autograd.grad(gv, params, allow_unused=True)
                trace_est = sum((vi * h).sum().item() for vi, h in zip(v, hvp) if h is not None)
                traces.append(trace_est)
    return float(np.mean(traces)) if traces else 0.0


def compute_fisher_trace(model, loader, device, loss_fn, max_batches=10):
    model.train()
    traces = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        t = sum(p.grad.data.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        traces.append(t)
    return float(np.mean(traces)) if traces else 0.0


def compute_gradient_norm(model, loader, device, loss_fn, max_batches=5):
    model.train()
    norms = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        n = math.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None))
        norms.append(n)
    return float(np.mean(norms)) if norms else 0.0


def compute_spectral_norm_approx(model, loader, device, loss_fn, n_iter=10, max_batches=2):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = math.sqrt(sum((vi ** 2).sum().item() for vi in v))
    v = [vi / v_norm for vi in v]

    lam = 0.0
    with _force_math_sdpa():
        for _ in range(n_iter):
            total_hvp = [torch.zeros_like(p) for p in params]
            count = 0
            for bi, (x, y) in enumerate(loader):
                if bi >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                loss = loss_fn(model(x), y)
                grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
                gv = sum((g * vi).sum() for g, vi in zip(grads, v) if g is not None)
                hvp = torch.autograd.grad(gv, params, allow_unused=True)
                for j, h in enumerate(hvp):
                    if h is not None:
                        total_hvp[j] += h.detach()
                count += 1
            if count > 0:
                total_hvp = [h / count for h in total_hvp]
            lam = math.sqrt(sum((h ** 2).sum().item() for h in total_hvp))
            if lam > 1e-10:
                v = [h / lam for h in total_hvp]
    return lam


def measure_intrinsic_curvature(model_factory, tasks, seed, device, epochs=4, n_hutch=10, n_fisher_batches=10):
    set_seed(seed)
    nc = tasks[0]['num_classes']
    model = model_factory(nc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in tasks[0]['train_loader']:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    ht = compute_hessian_trace(model, tasks[0]['train_loader'], device, loss_fn, n_samples=n_hutch, max_batches=3)
    ft = compute_fisher_trace(model, tasks[0]['train_loader'], device, loss_fn, max_batches=n_fisher_batches)
    gn = compute_gradient_norm(model, tasks[0]['train_loader'], device, loss_fn)
    sn = compute_spectral_norm_approx(model, tasks[0]['train_loader'], device, loss_fn, n_iter=10, max_batches=2)
    acc = evaluate(model, tasks[0]['test_loader'], device)
    n_params = sum(p.numel() for p in model.parameters())
    d_eff = ht / max(sn, 1e-10) if sn > 1e-10 else float(n_params)

    return {
        'hessian_trace': ht, 'fisher_trace': ft, 'gradient_norm': gn,
        'spectral_norm': sn, 'd_eff': d_eff, 'n_params': n_params, 'task0_accuracy': acc,
    }
