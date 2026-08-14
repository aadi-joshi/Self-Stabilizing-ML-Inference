"""
CIFAR-10/100 continual-learning task construction.

Addresses two gaps flagged in the external review of the preprint:
  1. Task-incremental vs class-incremental ambiguity (NEXT.md Sec 5.1): the
     original code silently implemented task-incremental learning (a shared
     head sized to classes-per-task, oracle task ID used to select the eval
     loader and to remap labels locally). `class_incremental=True` here
     instead uses ONE shared head sized to the full label space and does
     NOT remap labels, so forgetting is measured under the harder protocol.
  2. Task ordering / class-to-task split robustness (NEXT.md Sec 5.2):
     `task_order` permutes which classes are grouped into which task and in
     what sequence, so results can be checked across >=3 random assignments.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

_CACHE = {}


def _load_raw(dataset_name, data_root):
    key = dataset_name
    if key in _CACHE:
        return _CACHE[key]
    from torchvision import datasets
    if dataset_name == 'cifar10':
        train_d = datasets.CIFAR10(data_root, train=True, download=True)
        test_d = datasets.CIFAR10(data_root, train=False, download=True)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        n_classes = 10
    elif dataset_name == 'cifar100':
        train_d = datasets.CIFAR100(data_root, train=True, download=True)
        test_d = datasets.CIFAR100(data_root, train=False, download=True)
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        n_classes = 100
    else:
        raise ValueError(dataset_name)

    trx = (torch.tensor(train_d.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 - mean) / std
    try_ = torch.tensor(train_d.targets, dtype=torch.long)
    tex = (torch.tensor(test_d.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 - mean) / std
    tey = torch.tensor(test_d.targets, dtype=torch.long)
    _CACHE[key] = (trx, try_, tex, tey, n_classes)
    return _CACHE[key]


def build_tasks(dataset_name='cifar10', classes_per_task=2, max_per_class=1000,
                 batch_size=256, data_root='./data', task_order=None,
                 total_classes=None, class_incremental=False, device=None):
    """
    task_order: optional list of class indices (a permutation of
        range(total_classes)) specifying which classes are used and in what
        order they are grouped into tasks. If None, uses range(total_classes)
        in the canonical 0..C-1 order (matches original paper behavior).
    total_classes: how many of the dataset's classes to use in total (e.g.
        20 of CIFAR-100's 100, for classes-per-task sweeps that hold the
        label pool fixed while varying task granularity). Defaults to the
        full label space for the dataset.
    class_incremental: if True, all tasks share ONE head of size
        `total_classes` and labels are NOT remapped locally -- the model
        must distinguish all classes seen so far without an oracle task ID.
        If False (default, matches original paper), each task uses a local
        head of size `classes_per_task` with locally remapped labels
        (task-incremental; oracle task ID used at eval time).
    """
    trx, try_, tex, tey, n_classes_full = _load_raw(dataset_name, data_root)
    total_classes = total_classes or n_classes_full
    assert total_classes % classes_per_task == 0, \
        f"total_classes={total_classes} not divisible by classes_per_task={classes_per_task}"
    n_tasks = total_classes // classes_per_task

    order = list(task_order) if task_order is not None else list(range(total_classes))
    assert len(order) == total_classes

    tasks = []
    for t in range(n_tasks):
        classes = order[t * classes_per_task:(t + 1) * classes_per_task]
        cmap = {c: i for i, c in enumerate(classes)}  # local remap (task-incremental only)
        trm = sum(try_ == c for c in classes).bool()
        tem = sum(tey == c for c in classes).bool()
        tx, ty_o = trx[trm], try_[trm]
        ex, ey_o = tex[tem], tey[tem]
        if max_per_class and tx.shape[0] > max_per_class * classes_per_task:
            idx = torch.randperm(tx.shape[0])[:max_per_class * classes_per_task]
            tx, ty_o = tx[idx], ty_o[idx]

        if class_incremental:
            ty, ey = ty_o.clone(), ey_o.clone()  # keep global label ids, shared head
            head_size = total_classes
        else:
            ty = torch.zeros_like(ty_o)
            ey = torch.zeros_like(ey_o)
            for oc, nc in cmap.items():
                ty[ty_o == oc] = nc
                ey[ey_o == oc] = nc
            head_size = classes_per_task

        tasks.append({
            'train_loader': DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True),
            'test_loader': DataLoader(TensorDataset(ex, ey), batch_size=512),
            'train_x': tx, 'train_y': ty,
            'classes': classes, 'task_id': t,
            'num_classes': head_size,          # model output dim for this protocol
            'local_classes': classes_per_task,  # classes actually active in this task
            'class_incremental': class_incremental,
        })
    return tasks


def random_task_order(seed, total_classes=10):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(total_classes, generator=g).tolist()
