# ============================================================================
# Additional Continual Learning Baselines
#
# Implements:
#   1. Synaptic Intelligence (SI) — Zenke et al., 2017
#   2. Learning without Forgetting (LwF) — Li & Hoiem, 2016
#   3. Fixed Distillation — Knowledge distillation with fixed λ
#   4. Experience Replay — Small-buffer replay baseline
#
# Each is implemented as a composable regularizer that can be plugged into
# any standard training loop, matching the interface of EWCRegularizer.
# ============================================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from torch.utils.data import DataLoader, TensorDataset


class SynapticIntelligence:
    """
    Synaptic Intelligence (Zenke et al., ICML 2017).
    
    Tracks online importance of each parameter via the contribution
    to the total loss change along the optimization trajectory:
    
        Ω_k = Σ_t (∂L/∂θ_k · Δθ_k) / (Δθ_k² + ξ)
    
    Regularizer:
        L_SI = (c/2) Σ_k Ω_k (θ_k - θ*_k)²
    
    Key difference from EWC: SI computes importance *online* during 
    training rather than post-hoc via Fisher information. This makes
    it more reflective of the actual optimization trajectory.
    """

    def __init__(self, model: nn.Module, si_c: float = 1.0, xi: float = 1e-3):
        """
        Args:
            model: The neural network
            si_c: Regularization strength (analogous to EWC's λ)
            xi: Damping constant to prevent division by zero
        """
        self.si_c = si_c
        self.xi = xi

        # Initialize tracking variables
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Online tracking accumulators
        self._prev_params: Dict[str, torch.Tensor] = {}
        self._running_importance: Dict[str, torch.Tensor] = {}
        self._param_delta: Dict[str, torch.Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                self._prev_params[n] = p.data.clone()
                self._running_importance[n] = torch.zeros_like(p.data)
                self._param_delta[n] = torch.zeros_like(p.data)

    def update_running_importance(self, model: nn.Module):
        """
        Call after each optimizer step to accumulate importance.
        
        Tracks: w_k += -grad_k * Δθ_k  (path integral of loss gradient)
        """
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._prev_params:
                delta = p.data - self._prev_params[n]
                if p.grad is not None:
                    # Importance = negative gradient · parameter change
                    # This approximates the contribution to loss reduction
                    self._running_importance[n] += (-p.grad.data * delta)
                self._param_delta[n] += delta.abs()
                self._prev_params[n] = p.data.clone()

    def consolidate(self, model: nn.Module):
        """
        Call after finishing a task to consolidate importance scores.
        
        Normalizes running importance by total parameter change + damping:
            Ω_k = w_k / (Δθ_k² + ξ)
        """
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._running_importance:
                # Compute normalized importance
                new_omega = self._running_importance[n] / (
                    self._param_delta[n].pow(2) + self.xi
                )
                new_omega = torch.clamp(new_omega, min=0)  # Importance must be non-negative

                # Accumulate across tasks
                if n in self.omega:
                    self.omega[n] = self.omega[n] + new_omega
                else:
                    self.omega[n] = new_omega

                # Save current parameters as reference
                self.saved_params[n] = p.data.clone()

                # Reset accumulators for next task
                self._running_importance[n] = torch.zeros_like(p.data)
                self._param_delta[n] = torch.zeros_like(p.data)
                self._prev_params[n] = p.data.clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute SI regularization penalty.
        
        L_SI = (c/2) Σ_k Ω_k (θ_k - θ*_k)²
        """
        if not self.omega:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.omega:
                loss += (self.omega[n] * (p - self.saved_params[n]).pow(2)).sum()
        return 0.5 * self.si_c * loss


class LearningWithoutForgetting:
    """
    Learning without Forgetting (Li & Hoiem, ECCV 2016).
    
    Uses knowledge distillation from the model's own previous outputs
    to prevent forgetting. Before training on a new task, we record
    the soft outputs (logits) on the new task's data, then add a 
    distillation loss during training:
    
        L_LwF = L_task + α · KL(σ(z_old/T) || σ(z_new/T))
    
    where T is the temperature and σ is softmax.
    
    Key advantage: Does not require storing any old task data.
    Key limitation: Distillation targets come from the NEW task's 
    data distribution, which may not represent old tasks well.
    """

    def __init__(
        self,
        model: nn.Module,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
    ):
        """
        Args:
            model: The neural network
            lwf_alpha: Distillation loss weight
            temperature: Softmax temperature for distillation
        """
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.old_model: Optional[nn.Module] = None

    def begin_new_task(self, model: nn.Module):
        """
        Call before training on a new task.
        Saves a frozen copy of the current model for distillation.
        """
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def distillation_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LwF distillation loss.
        
        KL(softmax(old_logits/T) || softmax(new_logits/T)) * T²
        """
        if self.old_model is None:
            return torch.tensor(0.0, device=x.device)

        # Get current model outputs
        current_output = model(x)
        if isinstance(current_output, tuple):
            current_output = current_output[0]

        # Get old model outputs
        with torch.no_grad():
            old_output = self.old_model(x)
            if isinstance(old_output, tuple):
                old_output = old_output[0]

        # Distillation loss with temperature scaling
        T = self.temperature
        old_probs = F.softmax(old_output / T, dim=-1)
        new_log_probs = F.log_softmax(current_output / T, dim=-1)

        # KL divergence scaled by T²
        kl = F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (T ** 2)
        return self.lwf_alpha * kl


class FixedDistillation:
    """
    Fixed-coefficient knowledge distillation baseline.
    
    Unlike LwF which uses adaptive temperature, this uses a fixed
    MSE loss between old and new outputs:
    
        L_distill = λ · ||f_θ(x) - f_θ_old(x)||²
    
    This is essentially FTR without the adaptive Lagrangian mechanism,
    serving as an ablation to isolate the contribution of adaptive λ.
    """

    def __init__(self, model: nn.Module, distill_lambda: float = 1.0):
        self.distill_lambda = distill_lambda
        self.old_model: Optional[nn.Module] = None

    def begin_new_task(self, model: nn.Module):
        """Save frozen copy of model before new task training."""
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def penalty(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Compute fixed distillation penalty."""
        if self.old_model is None:
            return torch.tensor(0.0, device=x.device)

        current_output = model(x)
        if isinstance(current_output, tuple):
            current_output = current_output[0]

        with torch.no_grad():
            old_output = self.old_model(x)
            if isinstance(old_output, tuple):
                old_output = old_output[0]

        return self.distill_lambda * F.mse_loss(current_output, old_output)


class ExperienceReplay:
    """
    Small-buffer experience replay baseline.
    
    Stores a fixed-size buffer of (input, target) pairs from previous 
    tasks. During training on a new task, randomly samples from the 
    buffer and adds the replay loss to the current task loss:
    
        L_total = L_task + L_replay
    
    Buffer management: reservoir sampling to maintain a representative
    sample across all seen tasks.
    """

    def __init__(
        self,
        buffer_size: int = 500,
        replay_batch_size: int = 32,
        replay_weight: float = 1.0,
    ):
        """
        Args:
            buffer_size: Maximum number of examples to store
            replay_batch_size: Number of examples to replay per step
            replay_weight: Weight of replay loss relative to task loss
        """
        self.buffer_size = buffer_size
        self.replay_batch_size = replay_batch_size
        self.replay_weight = replay_weight
        
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.n_seen = 0

    def add_task_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_budget: Optional[int] = None,
    ):
        """
        Add data from a completed task to the replay buffer.
        Uses reservoir sampling to maintain uniform representation.
        
        Args:
            x: Input tensor (N, ...)
            y: Target tensor (N, ...)
            task_budget: Max samples per task (if None, uses buffer_size/n_tasks)
        """
        if task_budget is None:
            task_budget = self.buffer_size

        n = x.shape[0]
        indices = torch.randperm(n)[:min(n, task_budget)]
        
        for i in indices:
            self.n_seen += 1
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(x[i].cpu())
                self.buffer_y.append(y[i].cpu())
            else:
                # Reservoir sampling
                j = torch.randint(0, self.n_seen, (1,)).item()
                if j < self.buffer_size:
                    self.buffer_x[j] = x[i].cpu()
                    self.buffer_y[j] = y[i].cpu()

    def get_replay_batch(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch from the replay buffer."""
        if len(self.buffer_x) == 0:
            return None

        n = min(self.replay_batch_size, len(self.buffer_x))
        indices = torch.randperm(len(self.buffer_x))[:n]
        
        batch_x = torch.stack([self.buffer_x[i] for i in indices]).to(device)
        batch_y = torch.stack([self.buffer_y[i] for i in indices]).to(device)
        return batch_x, batch_y

    def replay_loss(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute loss on replay buffer samples."""
        batch = self.get_replay_batch(device)
        if batch is None:
            return torch.tensor(0.0, device=device)

        x, y = batch
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        return self.replay_weight * loss_fn(output, y)

    @property
    def size(self) -> int:
        return len(self.buffer_x)


class FeatureSpaceDrift:
    """
    Feature-space (intermediate representation) drift constraint.
    
    Instead of constraining output-space drift:
        D_f = E_x[||f_θ(x) - f_θ0(x)||²]
    
    This constrains the drift in an intermediate feature layer:
        D_h = E_x[||h_θ(x) - h_θ0(x)||²]
    
    where h_θ is the representation before the final classifier.
    
    This is an ablation variant to test whether constraining features
    is more or less effective than constraining outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        reference_data: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.reference_data = reference_data.to(device)
        
        # Store reference features
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        with torch.no_grad():
            if hasattr(ref_model, 'features'):
                self.ref_features = ref_model.features(self.reference_data)
            elif hasattr(ref_model, 'get_representations'):
                self.ref_features = ref_model.get_representations(self.reference_data)
            else:
                self.ref_features = ref_model(self.reference_data)
                if isinstance(self.ref_features, tuple):
                    self.ref_features = self.ref_features[0]
        del ref_model

    def compute(self, model: nn.Module, batch_size: int = 128) -> Dict[str, float]:
        """Compute feature-space drift (non-differentiable)."""
        model.eval()
        total_drift = 0.0
        n_points = 0

        n_total = self.reference_data.shape[0]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            x_batch = self.reference_data[start:end]
            ref_batch = self.ref_features[start:end]

            with torch.no_grad():
                if hasattr(model, 'features'):
                    curr = model.features(x_batch)
                elif hasattr(model, 'get_representations'):
                    curr = model.get_representations(x_batch)
                else:
                    curr = model(x_batch)
                    if isinstance(curr, tuple):
                        curr = curr[0]

                diff = (curr - ref_batch).pow(2).sum(dim=-1)
                total_drift += diff.sum().item()
            n_points += end - start

        model.train()
        return {
            'feature_drift': total_drift / max(n_points, 1),
        }

    def compute_differentiable(
        self, model: nn.Module, batch_size: int = 128
    ) -> torch.Tensor:
        """Compute feature-space drift (differentiable for training)."""
        total_drift = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_points = 0

        n_total = self.reference_data.shape[0]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            x_batch = self.reference_data[start:end]
            ref_batch = self.ref_features[start:end].detach()

            if hasattr(model, 'features'):
                curr = model.features(x_batch)
            elif hasattr(model, 'get_representations'):
                curr = model.get_representations(x_batch)
            else:
                curr = model(x_batch)
                if isinstance(curr, tuple):
                    curr = curr[0]

            diff = (curr - ref_batch).pow(2).sum(dim=-1).mean()
            total_drift = total_drift + diff * (end - start)
            n_points += end - start

        return total_drift / max(n_points, 1)
