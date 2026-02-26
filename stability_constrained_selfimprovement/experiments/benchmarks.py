# ============================================================================
# Extended Continual Learning Benchmarks
#
# Implements:
#   1. Split CIFAR-10 (5 binary tasks) — already in exp_continual.py
#   2. Split CIFAR-100 (10 or 20 tasks)
#   3. Permuted MNIST (10 permutations)
#   4. Rotated MNIST (rotations: 0°, 20°, 40°, ..., 180°)
#
# All benchmarks return a standardized list of task dicts compatible
# with the unified experiment runner.
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import List, Dict, Optional, Tuple
import os


def get_permuted_mnist_tasks(
    n_tasks: int = 10,
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
) -> List[Dict]:
    """
    Permuted MNIST benchmark.
    
    Each task applies a fixed random permutation to the 784 pixels.
    Task 0 uses the identity permutation (standard MNIST).
    All tasks share the same 10-class classification problem.
    
    This tests a model's ability to handle distribution shift while 
    retaining the same decision boundary structure.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Pre-load all data
    train_x = train_data.data.float().view(-1, 784) / 255.0
    train_x = (train_x - 0.1307) / 0.3081
    train_y = train_data.targets

    test_x = test_data.data.float().view(-1, 784) / 255.0
    test_x = (test_x - 0.1307) / 0.3081
    test_y = test_data.targets

    rng = np.random.RandomState(seed)
    tasks = []

    for task_id in range(n_tasks):
        if task_id == 0:
            perm = np.arange(784)
        else:
            perm = rng.permutation(784)

        perm_t = torch.LongTensor(perm)
        task_train_x = train_x[:, perm_t].unsqueeze(1).view(-1, 1, 28, 28)
        task_test_x = test_x[:, perm_t].unsqueeze(1).view(-1, 1, 28, 28)

        train_loader = DataLoader(
            TensorDataset(task_train_x, train_y),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        test_loader = DataLoader(
            TensorDataset(task_test_x, test_y),
            batch_size=batch_size, shuffle=False,
        )

        tasks.append({
            'train_loader': train_loader,
            'test_loader': test_loader,
            'classes': list(range(10)),
            'task_id': task_id,
            'train_x': task_train_x,
            'test_x': task_test_x,
            'num_classes': 10,
            'task_name': f'perm_{task_id}',
        })

    return tasks


def get_rotated_mnist_tasks(
    rotations: Optional[List[float]] = None,
    data_dir: str = "./data",
    batch_size: int = 128,
) -> List[Dict]:
    """
    Rotated MNIST benchmark.
    
    Each task rotates all images by a fixed angle.
    Default: 0°, 20°, 40°, 60°, 80°, 100°, 120°, 140°, 160°, 180°
    
    Tests smooth distribution shift tolerance.
    """
    if rotations is None:
        rotations = list(range(0, 200, 20))  # 10 tasks

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=base_transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=base_transform)

    # Pre-load
    train_x_raw = train_data.data.float().unsqueeze(1) / 255.0
    train_y = train_data.targets
    test_x_raw = test_data.data.float().unsqueeze(1) / 255.0
    test_y = test_data.targets

    tasks = []
    for task_id, angle in enumerate(rotations):
        # Apply rotation using grid_sample
        if angle == 0:
            task_train_x = (train_x_raw - 0.1307) / 0.3081
            task_test_x = (test_x_raw - 0.1307) / 0.3081
        else:
            task_train_x = _rotate_images(train_x_raw, angle)
            task_train_x = (task_train_x - 0.1307) / 0.3081
            task_test_x = _rotate_images(test_x_raw, angle)
            task_test_x = (task_test_x - 0.1307) / 0.3081

        train_loader = DataLoader(
            TensorDataset(task_train_x, train_y),
            batch_size=batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(task_test_x, test_y),
            batch_size=batch_size, shuffle=False,
        )

        tasks.append({
            'train_loader': train_loader,
            'test_loader': test_loader,
            'classes': list(range(10)),
            'task_id': task_id,
            'train_x': task_train_x,
            'test_x': task_test_x,
            'num_classes': 10,
            'task_name': f'rot_{angle}',
        })

    return tasks


def _rotate_images(images: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a batch of images by a given angle using affine transformation."""
    import math
    theta_rad = math.radians(angle)
    cos_a = math.cos(theta_rad)
    sin_a = math.sin(theta_rad)

    # Rotation matrix (2D affine)
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
    ], dtype=torch.float32).unsqueeze(0)

    # Process in batches to avoid memory issues
    batch_size = 1000
    results = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size]
        n = batch.shape[0]
        theta_batch = theta.expand(n, -1, -1)
        grid = torch.nn.functional.affine_grid(theta_batch, batch.size(), align_corners=False)
        rotated = torch.nn.functional.grid_sample(batch, grid, align_corners=False, padding_mode='zeros')
        results.append(rotated)

    return torch.cat(results, dim=0)


def get_split_cifar100_tasks(
    n_tasks: int = 10,
    data_dir: str = "./data",
    batch_size: int = 128,
) -> List[Dict]:
    """
    Split CIFAR-100 benchmark.
    
    Splits 100 classes into n_tasks disjoint groups.
    Default: 10 tasks × 10 classes each.
    
    More challenging than CIFAR-10 split due to:
    - More tasks = more forgetting opportunities
    - Finer-grained classes = harder to distinguish
    """
    # Download without transforms — load raw data in bulk for speed
    train_data = datasets.CIFAR100(data_dir, train=True, download=True)
    test_data = datasets.CIFAR100(data_dir, train=False, download=True)

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    all_train_x = torch.tensor(train_data.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    all_train_x = (all_train_x - mean) / std
    all_train_y = torch.tensor(train_data.targets, dtype=torch.long)

    all_test_x = torch.tensor(test_data.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    all_test_x = (all_test_x - mean) / std
    all_test_y = torch.tensor(test_data.targets, dtype=torch.long)

    classes_per_task = 100 // n_tasks
    tasks = []

    for task_id in range(n_tasks):
        task_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        class_map = {c: i for i, c in enumerate(task_classes)}

        # Filter using vectorized operations
        train_mask = torch.zeros(len(all_train_y), dtype=torch.bool)
        test_mask = torch.zeros(len(all_test_y), dtype=torch.bool)
        for c in task_classes:
            train_mask |= (all_train_y == c)
            test_mask |= (all_test_y == c)

        train_x = all_train_x[train_mask]
        train_y_orig = all_train_y[train_mask]
        test_x = all_test_x[test_mask]
        test_y_orig = all_test_y[test_mask]

        # Remap labels
        train_y = torch.zeros_like(train_y_orig)
        test_y = torch.zeros_like(test_y_orig)
        for orig_c, new_c in class_map.items():
            train_y[train_y_orig == orig_c] = new_c
            test_y[test_y_orig == orig_c] = new_c

        train_loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False,
        )

        tasks.append({
            'train_loader': train_loader,
            'test_loader': test_loader,
            'classes': task_classes,
            'task_id': task_id,
            'train_x': train_x,
            'test_x': test_x,
            'num_classes': classes_per_task,
            'task_name': f'cifar100_task{task_id}',
        })

    return tasks


class MNISTResNet(nn.Module):
    """
    Small ResNet adapted for MNIST (1-channel, 28x28).
    Uses the same BasicBlock structure as the CIFAR ResNet but with 
    adjusted initial convolution for grayscale input.
    """

    def __init__(self, num_classes: int = 10, base_width: int = 16):
        super().__init__()
        from models.resnet import BasicBlock

        self.in_planes = base_width
        self.conv1 = nn.Conv2d(1, base_width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        self.layer1 = self._make_layer(BasicBlock, base_width, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_width * 2, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_width * 4, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 4, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        from models.resnet import BasicBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))
