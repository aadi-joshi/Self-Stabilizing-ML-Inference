# ============================================================================
# Stability-Constrained Optimizer
#
# Implements the constrained optimization:
#   θ_{t+1} = argmin_θ L(θ)  s.t.  D_f(θ, θ_0) ≤ ε_t
#
# Via the Lagrangian relaxation:
#   L_total = L_task + λ · D_f(θ, θ_0)
#
# With dual gradient ascent on λ:
#   λ_{t+1} = max(0, λ_t + η_λ · (D_f(θ_t, θ_0) - ε_t))
#
# This converts the hard constraint into a soft penalty with an adaptively
# tuned coefficient, yielding an approximate saddle-point solution.
# ============================================================================

import copy
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from .functional_drift import FunctionalDrift


class StabilityConstrainedOptimizer:
    """
    Wraps a standard optimizer with a functional drift constraint.
    
    At each step:
        1. Compute task loss L_task
        2. Compute functional drift D_f(θ, θ_0)
        3. Form Lagrangian: L_total = L_task + λ · D_f
        4. Backprop and step the wrapped optimizer
        5. Update λ via dual ascent: λ ← max(0, λ + η_λ(D_f - ε))
        
    Theoretical Justification:
        Under convexity of D_f w.r.t. θ and bounded gradients, the 
        primal-dual iterates converge to an ε-approximate KKT point 
        at rate O(1/√T). Even for non-convex losses, the Lagrangian 
        relaxation provides a principled soft constraint that empirically 
        controls drift while allowing task optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        drift_module: FunctionalDrift,
        lambda_init: float = 0.1,
        lambda_lr: float = 0.01,
        lambda_max: float = 100.0,
        epsilon_scheduler: Optional['EpsilonScheduler'] = None,
        epsilon_fixed: float = 1.0,
        grad_clip: float = 1.0,
        activation_step: int = 0,
    ):
        """
        Args:
            model: The model being trained
            base_optimizer: Wrapped optimizer (Adam, SGD, etc.)
            drift_module: FunctionalDrift instance for computing D_f
            lambda_init: Initial Lagrange multiplier
            lambda_lr: Learning rate for dual variable update
            lambda_max: Maximum value for λ (prevents instability)
            epsilon_scheduler: Adaptive ε schedule (if None, uses fixed ε)
            epsilon_fixed: Fixed ε value (used if scheduler is None)
            grad_clip: Maximum gradient norm
            activation_step: Step at which to activate the constraint
        """
        self.model = model
        self.optimizer = base_optimizer
        self.drift_module = drift_module

        self.lambda_val = lambda_init
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max

        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_fixed = epsilon_fixed
        self.epsilon = epsilon_fixed

        self.grad_clip = grad_clip
        self.activation_step = activation_step

        self.step_count = 0
        self.history: Dict[str, list] = {
            "lambda": [],
            "epsilon": [],
            "drift": [],
            "task_loss": [],
            "total_loss": [],
            "constraint_violation": [],
        }

    def step(
        self,
        task_loss: torch.Tensor,
        drift_batch_size: int = 128,
    ) -> Dict[str, float]:
        """
        Perform one optimization step with functional drift constraint.
        
        Args:
            task_loss: The task-specific loss (already computed, with grad graph)
            drift_batch_size: Batch size for drift estimation
            
        Returns:
            Dict with step metrics
        """
        self.step_count += 1

        # Update epsilon from scheduler
        if self.epsilon_scheduler is not None:
            self.epsilon = self.epsilon_scheduler.get_epsilon(self.step_count)
        else:
            self.epsilon = self.epsilon_fixed

        # Check if constraint is active
        constraint_active = self.step_count >= self.activation_step

        if constraint_active and self.lambda_val > 0:
            # Compute differentiable drift
            drift_loss = self.drift_module.compute_differentiable(
                self.model, batch_size=drift_batch_size
            )

            # Form Lagrangian: L_total = L_task + λ · D_f
            total_loss = task_loss + self.lambda_val * drift_loss

            # Compute drift value for logging and dual update
            drift_val = drift_loss.item()
        else:
            total_loss = task_loss
            drift_val = 0.0

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step (primal update)
        self.optimizer.step()

        # Dual variable update: λ ← max(0, λ + η_λ(D_f - ε))
        if constraint_active:
            constraint_violation = drift_val - self.epsilon
            self.lambda_val = max(0.0, min(
                self.lambda_val + self.lambda_lr * constraint_violation,
                self.lambda_max
            ))
        else:
            constraint_violation = 0.0

        # Record history
        metrics = {
            "lambda": self.lambda_val,
            "epsilon": self.epsilon,
            "drift": drift_val,
            "task_loss": task_loss.item(),
            "total_loss": total_loss.item(),
            "constraint_violation": constraint_violation,
        }
        for key, val in metrics.items():
            self.history[key].append(val)

        return metrics

    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "lambda_val": self.lambda_val,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "history": self.history,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.lambda_val = state["lambda_val"]
        self.epsilon = state["epsilon"]
        self.step_count = state["step_count"]
        self.history = state["history"]


class EpsilonScheduler:
    """
    Adaptive epsilon scheduler for the drift constraint budget.
    
    Schedules:
        - 'fixed': Constant ε
        - 'linear_decay': Linear decay from ε_init to ε_min
        - 'cosine': Cosine annealing from ε_init to ε_min
        - 'uncertainty': Adaptive ε based on model uncertainty estimation
          
    The uncertainty-based schedule is the novel contribution:
        ε_t = ε_base · (1 + scale · H(p_θ(y|x)))
    where H is the predictive entropy. High uncertainty → larger ε (more 
    freedom to change), low uncertainty → tighter constraint (preserve).
    """

    def __init__(
        self,
        schedule_type: str = "fixed",
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_max: float = 10.0,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        decay_rate: float = 0.995,
        uncertainty_scale: float = 1.0,
        uncertainty_fn: Optional[Callable] = None,
    ):
        self.schedule_type = schedule_type
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_rate = decay_rate
        self.uncertainty_scale = uncertainty_scale
        self.uncertainty_fn = uncertainty_fn

        self._current_uncertainty = 0.0

    def update_uncertainty(self, uncertainty: float):
        """Update current uncertainty estimate (for uncertainty schedule)."""
        self._current_uncertainty = uncertainty

    def get_epsilon(self, step: int) -> float:
        """Get epsilon for current step."""
        if step < self.warmup_steps:
            # During warmup, use large epsilon (no constraint)
            return self.epsilon_max

        effective_step = step - self.warmup_steps
        effective_total = max(self.total_steps - self.warmup_steps, 1)

        if self.schedule_type == "fixed":
            eps = self.epsilon_init

        elif self.schedule_type == "linear_decay":
            progress = min(effective_step / effective_total, 1.0)
            eps = self.epsilon_init + (self.epsilon_min - self.epsilon_init) * progress

        elif self.schedule_type == "cosine":
            progress = min(effective_step / effective_total, 1.0)
            eps = self.epsilon_min + 0.5 * (self.epsilon_init - self.epsilon_min) * (
                1 + math.cos(math.pi * progress)
            )

        elif self.schedule_type == "exponential":
            eps = max(self.epsilon_init * (self.decay_rate ** effective_step), self.epsilon_min)

        elif self.schedule_type == "uncertainty":
            # Novel: ε_t = ε_base · (1 + scale · uncertainty)
            base = self.epsilon_init * (self.decay_rate ** effective_step)
            base = max(base, self.epsilon_min)
            eps = base * (1.0 + self.uncertainty_scale * self._current_uncertainty)
            eps = min(eps, self.epsilon_max)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return max(eps, self.epsilon_min)


class EWCRegularizer:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017) baseline.
    
    L_EWC = L_task + (λ/2) Σ_i F_i (θ_i - θ*_i)²
    
    where F_i is the diagonal of the Fisher information matrix and θ*
    are the parameters after training on previous tasks.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.ewc_lambda = ewc_lambda
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}

    def estimate_fisher(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_samples: int = 1000,
    ):
        """Estimate diagonal Fisher information after task completion."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0

        for x, y in data_loader:
            if count >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            # Use log-likelihood of true labels
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * x.shape[0]
            count += x.shape[0]

        # Normalize
        for n in fisher:
            fisher[n] /= max(count, 1)

        self.fisher_diag = fisher
        self.saved_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty: (λ/2) Σ_i F_i (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher_diag:
                loss += (self.fisher_diag[n] * (p - self.saved_params[n]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * loss
