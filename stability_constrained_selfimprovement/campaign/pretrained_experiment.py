#!/usr/bin/env python3
"""
Modern-relevance stretch experiment: a pretrained ViT-B/16 backbone (ImageNet1k
weights) adapted with LoRA on Split CIFAR-100 (10 tasks x 10 classes),
comparing FTR against LwF, EWC, and vanilla fine-tuning in a setting closer
to how continual learning is actually done today (pretrained backbone +
parameter-efficient adaptation) rather than training small CNNs from
scratch. This is a standalone, self-contained module: different data
pipeline (224x224 images, ImageNet normalization) and different model
family (LoRA-wrapped torchvision ViT) from the rest of the campaign, so it
does not share `campaign.data` / `campaign.models`, but reuses the same
method logic (FTR/LwF/EWC dual-ascent and distillation update rules) for
direct comparability with the rest of the paper.
"""
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# LoRA wrapper for nn.Linear
# ======================================================================
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update:
    y = W_0 x + b_0 + (alpha/r) * B A x, A in R^{r x in}, B in R^{out x r}."""

    def __init__(self, base_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f) * (1.0 / math.sqrt(in_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A.t()) @ self.B.t() * self.scale
        return base_out + lora_out


def apply_lora_to_vit(vit, r=8, alpha=16):
    """Replaces the nn.Linear layers inside each ViT block's MLP with
    LoRALinear wrappers, freezing everything else including attention.
    torchvision's nn.MultiheadAttention reads `in_proj_weight`/`out_proj`
    directly inside a fused functional call rather than invoking
    `out_proj.forward()`, so a module-wrapping approach (as used here for
    the MLP) cannot target attention without reimplementing the attention
    forward pass; MLP-only LoRA is standard practice and still gives a
    genuinely parameter-efficient adaptation. Returns list of trainable
    LoRA parameters."""
    for block in vit.encoder.layers:
        mlp = block.mlp
        for i, layer in enumerate(mlp):
            if isinstance(layer, nn.Linear):
                mlp[i] = LoRALinear(layer, r=r, alpha=alpha)
    lora_params = [p for n, p in vit.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    return lora_params


def build_pretrained_lora_model(num_classes, r=8, alpha=16, device='cuda'):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights)
    for p in vit.parameters():
        p.requires_grad = False
    lora_params = apply_lora_to_vit(vit, r=r, alpha=alpha)
    in_features = vit.heads.head.in_features
    vit.heads.head = nn.Linear(in_features, num_classes)
    head_params = list(vit.heads.head.parameters())
    for p in head_params:
        p.requires_grad = True
    vit = vit.to(device)
    trainable = lora_params + head_params
    return vit, trainable, weights.transforms()


# ======================================================================
# Data: Split CIFAR-100, 10 tasks x 10 classes, resized to 224x224
# ======================================================================
def build_split_cifar100_pretrained(n_tasks=10, max_per_class=200, batch_size=64,
                                     data_root='./data'):
    from torchvision import datasets
    from torch.utils.data import DataLoader, TensorDataset

    train_d = datasets.CIFAR100(data_root, train=True, download=True)
    test_d = datasets.CIFAR100(data_root, train=False, download=True)

    # ImageNet normalization; upsample 32x32 -> 224x224 with simple nearest/
    # bilinear interpolation done on the fly via F.interpolate in the loader
    # wrapper below to avoid materializing 224x224 uint8 arrays in RAM for
    # the whole dataset.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    trx = torch.tensor(train_d.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    try_ = torch.tensor(train_d.targets, dtype=torch.long)
    tex = torch.tensor(test_d.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    tey = torch.tensor(test_d.targets, dtype=torch.long)

    cpt = 100 // n_tasks
    tasks = []
    for t in range(n_tasks):
        classes = list(range(t * cpt, (t + 1) * cpt))
        cmap = {c: i for i, c in enumerate(classes)}
        trm = sum(try_ == c for c in classes).bool()
        tem = sum(tey == c for c in classes).bool()
        tx, ty_o = trx[trm], try_[trm]
        ex, ey_o = tex[tem], tey[tem]
        if max_per_class and tx.shape[0] > max_per_class * cpt:
            idx = torch.randperm(tx.shape[0])[:max_per_class * cpt]
            tx, ty_o = tx[idx], ty_o[idx]
        ty = torch.zeros_like(ty_o)
        ey = torch.zeros_like(ey_o)
        for oc, nc in cmap.items():
            ty[ty_o == oc] = nc
            ey[ey_o == oc] = nc

        class ResizeNormDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x, self.y = x, y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, i):
                img = F.interpolate(self.x[i:i + 1], size=(224, 224), mode='bilinear',
                                     align_corners=False)[0]
                img = (img - mean) / std
                return img, self.y[i]

        tasks.append({
            'train_loader': DataLoader(ResizeNormDataset(tx, ty), batch_size=batch_size, shuffle=True, num_workers=2),
            'test_loader': DataLoader(ResizeNormDataset(ex, ey), batch_size=128, num_workers=2),
            'classes': classes, 'task_id': t, 'num_classes': cpt,
        })
    return tasks


# ======================================================================
# Training engine (FTR / LwF / EWC / finetune), mirrors campaign.engine's
# update rules for direct comparability, adapted for the LoRA parameter set.
# ======================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def run_pretrained_experiment(method, seed, device, n_tasks=10, epochs_per_task=2,
                               method_cfg=None, r=8, alpha=16, data_root='./data'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    method_cfg = method_cfg or {}
    tasks = build_split_cifar100_pretrained(n_tasks=n_tasks, data_root=data_root)
    nc = tasks[0]['num_classes']
    model, trainable_params, _ = build_pretrained_lora_model(nc, r=r, alpha=alpha, device=device)
    optimizer = torch.optim.Adam(trainable_params, lr=method_cfg.get('lr', 1e-3))
    loss_fn = nn.CrossEntropyLoss()

    old_model = None
    ewc_fisher, ewc_params = {}, {}
    n_tasks_run = len(tasks)
    acc_matrix = np.zeros((n_tasks_run, n_tasks_run))

    eps = method_cfg.get('epsilon', 5.0)
    lam_init = method_cfg.get('lambda_init', 1.0)
    lam_lr = method_cfg.get('lambda_lr', 0.005)
    lam_max = method_cfg.get('lambda_max', 50.0)
    momentum = method_cfg.get('lambda_momentum', 0.9)
    temp = method_cfg.get('temperature', 2.0)
    warmup_ep = method_cfg.get('warmup_epochs', 1)

    for task_id in range(n_tasks_run):
        task = tasks[task_id]
        if task_id > 0 and method in ('lwf', 'ftr'):
            old_model = copy.deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        if task_id > 0 and method == 'ftr':
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
                        if n in ewc_fisher and p.requires_grad:
                            reg_loss = reg_loss + (ewc_fisher[n] * (p - ewc_params[n]).pow(2)).sum()
                    reg_loss = method_cfg.get('ewc_lambda', 1000.0) * reg_loss
                elif method == 'lwf' and task_id > 0 and old_model is not None:
                    with torch.no_grad():
                        old_out = old_model(x)
                    alpha_lwf = method_cfg.get('lwf_alpha', 0.7)
                    old_soft = F.softmax(old_out / temp, dim=-1)
                    new_log = F.log_softmax(output / temp, dim=-1)
                    reg_loss = alpha_lwf * temp * temp * F.kl_div(new_log, old_soft, reduction='batchmean')

                if method == 'ftr' and task_id > 0:
                    step_count += 1
                    with torch.no_grad():
                        old_out = old_model(x)
                    old_soft = F.softmax(old_out / temp, dim=-1)
                    new_log = F.log_softmax(output / temp, dim=-1)
                    dv = temp * temp * F.kl_div(new_log, old_soft, reduction='batchmean')
                    if step_count > wb:
                        total_loss = task_loss + lam * dv
                        viol = dv.item() - eps
                        ema_viol = momentum * ema_viol + (1 - momentum) * viol
                        lam = max(0.0, min(lam_max, lam + lam_lr * ema_viol))
                    else:
                        total_loss = task_loss + dv
                else:
                    total_loss = task_loss + reg_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

        if method == 'ewc':
            fisher = {}
            model.eval()
            for x, y in task['train_loader']:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
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

        model.eval()
        for eid in range(task_id + 1):
            acc_matrix[task_id, eid] = evaluate(model, tasks[eid]['test_loader'], device)

    aa = acc_matrix[n_tasks_run - 1, :n_tasks_run].mean()
    fgt_v = []
    for j in range(n_tasks_run - 1):
        best_j = max(acc_matrix[i, j] for i in range(j, n_tasks_run))
        fgt_v.append(max(0, best_j - acc_matrix[n_tasks_run - 1, j]))
    return {
        'average_accuracy': float(aa),
        'forgetting': float(np.mean(fgt_v)) if fgt_v else 0.0,
        'acc_matrix': acc_matrix.tolist(),
    }
