# ============================================================================
# Trainer Module: Unified training loop for all methods and experiments
# ============================================================================

import os
import sys
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Callable, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.common import (
    AverageMeter, param_distance, param_norm, grad_norm, flatten_params
)
from metrics.functional_drift import FunctionalDrift, RepresentationDrift
from metrics.constrained_optimizer import (
    StabilityConstrainedOptimizer, EpsilonScheduler, EWCRegularizer
)
from metrics.experiment_metrics import ExperimentMetrics


class BaseTrainer:
    """
    Base trainer with common logic for all methods.
    
    Methods supported:
        - 'baseline': Standard Adam/SGD training
        - 'weight_decay': Adam with weight decay
        - 'ewc': Elastic Weight Consolidation
        - 'functional_trust': Our method (functional drift constraint)
        - 'kl_trust': KL divergence trust region (for RL)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        method: str = "baseline",
        lr: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        seed: int = 42,
        experiment_name: str = "default",
        # Functional trust region params
        drift_config: Optional[Dict] = None,
        epsilon_config: Optional[Dict] = None,
        # EWC params
        ewc_lambda: float = 1000.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.method = method
        self.grad_clip = grad_clip
        self.seed = seed

        # Store initial model for parameter drift measurement
        self.initial_model = copy.deepcopy(model).to(device)
        self.initial_model.eval()
        for p in self.initial_model.parameters():
            p.requires_grad = False

        # Setup optimizer
        if method == "weight_decay":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay or 0.01
            )
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Functional drift module (initialized later when reference data is available)
        self.drift_module: Optional[FunctionalDrift] = None
        self.repr_drift: Optional[RepresentationDrift] = None
        self.constrained_optimizer: Optional[StabilityConstrainedOptimizer] = None
        self.drift_config = drift_config or {}
        self.epsilon_config = epsilon_config or {}

        # EWC
        self.ewc = EWCRegularizer(model, ewc_lambda) if method == "ewc" else None

        # Metrics
        self.metrics = ExperimentMetrics(
            seed=seed, method=method, experiment=experiment_name
        )
        self.global_step = 0

    def setup_drift_constraint(
        self,
        reference_data: torch.Tensor,
        total_steps: int = 10000,
    ):
        """Initialize functional drift module with reference data."""
        self.drift_module = FunctionalDrift(
            reference_model=self.model,
            reference_data=reference_data,
            norm_type=self.drift_config.get("norm_type", "l2"),
            device=self.device,
        )

        self.repr_drift = RepresentationDrift(
            reference_model=self.model,
            reference_data=reference_data,
            device=self.device,
        )

        if self.method == "functional_trust":
            epsilon_scheduler = EpsilonScheduler(
                schedule_type=self.epsilon_config.get("type", "fixed"),
                epsilon_init=self.drift_config.get("epsilon_init", 1.0),
                epsilon_min=self.drift_config.get("epsilon_min", 0.01),
                epsilon_max=self.drift_config.get("epsilon_max", 10.0),
                warmup_steps=self.epsilon_config.get("warmup_steps", 100),
                total_steps=total_steps,
                decay_rate=self.epsilon_config.get("decay_rate", 0.995),
                uncertainty_scale=self.epsilon_config.get("uncertainty_scale", 1.0),
            )

            self.constrained_optimizer = StabilityConstrainedOptimizer(
                model=self.model,
                base_optimizer=self.optimizer,
                drift_module=self.drift_module,
                lambda_init=self.drift_config.get("lambda_init", 0.1),
                lambda_lr=self.drift_config.get("lambda_lr", 0.01),
                lambda_max=self.drift_config.get("lambda_max", 100.0),
                epsilon_scheduler=epsilon_scheduler,
                epsilon_fixed=self.drift_config.get("epsilon_init", 1.0),
                grad_clip=self.grad_clip,
                activation_step=self.drift_config.get("activation_step", 0),
            )

    def train_step(self, batch, loss_fn: Callable) -> Dict[str, float]:
        """
        Single training step for supervised learning.
        
        Args:
            batch: (x, y) tuple
            loss_fn: Function(model_output, target) -> loss tensor
        """
        self.model.train()
        self.global_step += 1

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        output = self.model(x)
        task_loss = loss_fn(output, y)

        # Method-specific handling
        if self.method == "ewc" and self.ewc is not None:
            ewc_penalty = self.ewc.penalty(self.model)
            task_loss = task_loss + ewc_penalty

        if self.method == "functional_trust" and self.constrained_optimizer is not None:
            # Use constrained optimizer (handles backward + step internally)
            step_metrics = self.constrained_optimizer.step(task_loss)
        else:
            # Standard training
            self.optimizer.zero_grad()
            task_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            step_metrics = {"task_loss": task_loss.item(), "total_loss": task_loss.item()}

        # Compute additional metrics
        step_metrics["grad_norm"] = grad_norm(self.model)
        step_metrics["param_norm_drift"] = param_distance(self.model, self.initial_model)

        if self.drift_module is not None:
            drift_info = self.drift_module.compute(self.model)
            step_metrics["functional_drift"] = drift_info["drift"]
            if "drift" not in step_metrics:
                step_metrics["drift"] = drift_info["drift"]

        # Accuracy (for classification)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() > 1 and output.shape[-1] > 1:
            pred = output.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            step_metrics["accuracy"] = acc

        self.metrics.log_step(self.global_step, step_metrics)
        return step_metrics

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        loss_fn: Optional[Callable] = None,
        task_name: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            if isinstance(output, tuple):
                output = output[0]

            if loss_fn is not None:
                loss = loss_fn(output, y)
                total_loss += loss.item() * x.shape[0]

            if output.dim() > 1 and output.shape[-1] > 1:
                pred = output.argmax(dim=-1)
                total_correct += (pred == y).sum().item()
            total_samples += x.shape[0]

        result = {
            f"{task_name}_loss": total_loss / max(total_samples, 1),
            f"{task_name}_accuracy": total_correct / max(total_samples, 1),
        }

        # Drift metrics
        if self.drift_module is not None:
            drift_info = self.drift_module.compute(self.model)
            result[f"{task_name}_drift"] = drift_info["drift"]

        if self.repr_drift is not None:
            repr_info = self.repr_drift.compute(self.model)
            result[f"{task_name}_cka"] = repr_info["cka_similarity"]

        return result

    def update_reference(self, new_reference_data: Optional[torch.Tensor] = None):
        """Update drift reference to current model (e.g., between tasks)."""
        if self.drift_module is not None:
            self.drift_module.update_reference(self.model, new_reference_data)
        if self.repr_drift is not None:
            self.repr_drift = RepresentationDrift(
                self.model, new_reference_data or self.repr_drift.reference_data, device=self.device
            )
