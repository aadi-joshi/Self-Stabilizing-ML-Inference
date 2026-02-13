# ============================================================================
# Utility Modules
# ============================================================================

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config_device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config_device)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep-merge two config dicts (override takes precedence)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def param_norm(model: torch.nn.Module) -> float:
    """Compute L2 norm of all parameters."""
    return flatten_params(model).norm(2).item()


def grad_norm(model: torch.nn.Module) -> float:
    """Compute L2 norm of all gradients."""
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm(2).item()


def param_distance(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """Compute L2 distance between two models' parameters."""
    pa = flatten_params(model_a)
    pb = flatten_params(model_b)
    return (pa - pb).norm(2).item()


class AverageMeter:
    """Computes and stores running average and standard deviation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.values = []

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))
