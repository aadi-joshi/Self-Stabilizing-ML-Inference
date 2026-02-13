# ============================================================================
# Functional Drift Computation Module
#
# Core contribution: measures drift in function space, not parameter space.
#
# Mathematical Foundation:
#   D_f(θ_t, θ_0) = E_{x ~ D} [ ||f_{θ_t}(x) - f_{θ_0}(x)||² ]
#
# This is estimated via Monte Carlo over a fixed reference set X_ref:
#   D̂_f(θ_t, θ_0) = (1/|X_ref|) Σ_i ||f_{θ_t}(x_i) - f_{θ_0}(x_i)||²
#
# The reference set is sampled once from the data distribution and kept fixed
# to ensure consistent measurement across training.
# ============================================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import numpy as np


class FunctionalDrift:
    """
    Computes functional drift between a current model and a reference model.
    
    Measures how much the function computed by the network has changed,
    independent of how the parameters themselves have changed.
    
    Key insight: Two models with very different parameters can compute
    the same function (parameter symmetries), and two models with similar
    parameters can compute very different functions (high curvature regions).
    Functional drift captures what matters: behavioral change.
    
    Supports multiple drift norms:
        - 'l2': E_x[||f_θ(x) - f_θ0(x)||²]  (default)
        - 'linf': E_x[max_i |f_θ(x)_i - f_θ0(x)_i|]
        - 'kl': E_x[KL(softmax(f_θ0(x)) || softmax(f_θ(x)))]
    """

    def __init__(
        self,
        reference_model: nn.Module,
        reference_data: torch.Tensor,
        norm_type: str = "l2",
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            reference_model: The baseline model θ_0 (will be deep-copied and frozen)
            reference_data: Fixed reference inputs X_ref for drift estimation
            norm_type: Type of function-space norm ('l2', 'linf', 'kl')
            device: Computation device
        """
        self.device = device
        self.norm_type = norm_type

        # Deep copy and freeze reference model
        self.reference_model = copy.deepcopy(reference_model).to(device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        # Store reference data
        self.reference_data = reference_data.to(device)

        # Pre-compute reference outputs for efficiency
        with torch.no_grad():
            self.reference_outputs = self._forward(self.reference_model, self.reference_data)

    def _forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass handling different model types."""
        model.eval()
        with torch.no_grad():
            output = model(x)
            # Handle tuple outputs (e.g., policy networks return (probs, values))
            if isinstance(output, tuple):
                output = output[0]
        return output

    def compute(
        self,
        current_model: nn.Module,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """
        Compute functional drift D_f(θ_t, θ_0).
        
        Returns dict with:
            - 'drift': The scalar drift value
            - 'drift_per_dim': Drift normalized by output dimension
            - 'max_drift': Maximum pointwise drift
        """
        current_model.eval()
        total_drift = 0.0
        max_drift = 0.0
        n_points = 0

        n_total = self.reference_data.shape[0]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            x_batch = self.reference_data[start:end]
            ref_batch = self.reference_outputs[start:end]

            with torch.no_grad():
                current_output = current_model(x_batch)
                if isinstance(current_output, tuple):
                    current_output = current_output[0]

            if self.norm_type == "l2":
                # ||f_θ(x) - f_θ0(x)||²
                diff = (current_output - ref_batch).pow(2).sum(dim=-1)
                total_drift += diff.sum().item()
                max_drift = max(max_drift, diff.max().item())

            elif self.norm_type == "linf":
                # max_i |f_θ(x)_i - f_θ0(x)_i|
                diff = (current_output - ref_batch).abs().max(dim=-1)[0]
                total_drift += diff.sum().item()
                max_drift = max(max_drift, diff.max().item())

            elif self.norm_type == "kl":
                # KL(softmax(f_θ0) || softmax(f_θ))
                p = F.log_softmax(ref_batch, dim=-1)
                q = F.log_softmax(current_output, dim=-1)
                kl = F.kl_div(q, p.exp(), reduction="none").sum(dim=-1)
                total_drift += kl.sum().item()
                max_drift = max(max_drift, kl.max().item())

            n_points += end - start

        current_model.train()

        mean_drift = total_drift / max(n_points, 1)
        output_dim = self.reference_outputs.shape[-1] if self.reference_outputs.dim() > 1 else 1

        return {
            "drift": mean_drift,
            "drift_per_dim": mean_drift / max(output_dim, 1),
            "max_drift": max_drift,
        }

    def compute_differentiable(
        self,
        current_model: nn.Module,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute functional drift as a differentiable tensor for gradient-based optimization.
        
        This is the key function used in the Lagrangian:
            L_total = L_task + λ · D_f(θ, θ_0)
        
        The gradient ∇_θ D_f flows through the current model's forward pass.
        """
        total_drift = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_points = 0

        n_total = self.reference_data.shape[0]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            x_batch = self.reference_data[start:end]
            ref_batch = self.reference_outputs[start:end].detach()

            current_output = current_model(x_batch)
            if isinstance(current_output, tuple):
                current_output = current_output[0]

            if self.norm_type == "l2":
                diff = (current_output - ref_batch).pow(2).sum(dim=-1).mean()
            elif self.norm_type == "linf":
                diff = (current_output - ref_batch).abs().max(dim=-1)[0].mean()
            elif self.norm_type == "kl":
                p = F.softmax(ref_batch, dim=-1)
                q = F.log_softmax(current_output, dim=-1)
                diff = F.kl_div(q, p, reduction="batchmean")
            else:
                raise ValueError(f"Unknown norm type: {self.norm_type}")

            total_drift = total_drift + diff * (end - start)
            n_points += end - start

        return total_drift / max(n_points, 1)

    def update_reference(self, new_model: nn.Module, new_data: Optional[torch.Tensor] = None):
        """
        Update the reference model and optionally the reference data.
        Used when transitioning between tasks in continual learning.
        """
        self.reference_model = copy.deepcopy(new_model).to(self.device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        if new_data is not None:
            self.reference_data = new_data.to(self.device)

        with torch.no_grad():
            self.reference_outputs = self._forward(self.reference_model, self.reference_data)


class RepresentationDrift:
    """
    Measures drift in intermediate representations using CCA/SVCCA-inspired metrics.
    
    Computes similarity between hidden representations of reference and current model
    using Centered Kernel Alignment (CKA), which is more tractable than full CCA.
    
    CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F · ||Y^T Y||_F)
    
    where X, Y are representation matrices (n_samples × n_features).
    """

    def __init__(
        self,
        reference_model: nn.Module,
        reference_data: torch.Tensor,
        representation_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.representation_fn = representation_fn

        # Get reference representations
        ref_model = copy.deepcopy(reference_model).to(device).eval()
        with torch.no_grad():
            if representation_fn is not None:
                self.ref_repr = representation_fn(ref_model, reference_data.to(device))
            elif hasattr(reference_model, "features"):
                self.ref_repr = ref_model.features(reference_data.to(device))
            elif hasattr(reference_model, "get_representations"):
                self.ref_repr = ref_model.get_representations(reference_data.to(device))
            else:
                self.ref_repr = ref_model(reference_data.to(device))
                if isinstance(self.ref_repr, tuple):
                    self.ref_repr = self.ref_repr[0]

        self.reference_data = reference_data.to(device)
        del ref_model

    @staticmethod
    def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute Linear CKA between two representation matrices.
        
        CKA measures functional similarity regardless of invertible linear transforms.
        """
        # Flatten if needed
        X = X.reshape(X.shape[0], -1).float()
        Y = Y.reshape(Y.shape[0], -1).float()

        # Center
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
        hsic_xx = (X @ X.T).pow(2).sum()
        hsic_yy = (Y @ Y.T).pow(2).sum()

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-10:
            return 0.0
        return (hsic_xy / denom).item()

    def compute(self, current_model: nn.Module) -> Dict[str, float]:
        """Compute representation similarity metrics."""
        current_model.eval()
        with torch.no_grad():
            if self.representation_fn is not None:
                curr_repr = self.representation_fn(current_model, self.reference_data)
            elif hasattr(current_model, "features"):
                curr_repr = current_model.features(self.reference_data)
            elif hasattr(current_model, "get_representations"):
                curr_repr = current_model.get_representations(self.reference_data)
            else:
                curr_repr = current_model(self.reference_data)
                if isinstance(curr_repr, tuple):
                    curr_repr = curr_repr[0]

        current_model.train()

        cka = self.linear_cka(self.ref_repr, curr_repr)

        # L2 representation drift
        ref_flat = self.ref_repr.reshape(self.ref_repr.shape[0], -1)
        curr_flat = curr_repr.reshape(curr_repr.shape[0], -1)
        repr_l2 = (ref_flat - curr_flat).pow(2).sum(dim=-1).mean().item()

        return {
            "cka_similarity": cka,
            "representation_drift_l2": repr_l2,
        }
