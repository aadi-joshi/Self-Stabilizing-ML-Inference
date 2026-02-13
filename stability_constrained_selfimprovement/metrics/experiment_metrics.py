# ============================================================================
# Comprehensive Metrics Module
# Tracks: task performance, drift, forgetting, stability, statistical rigor
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from scipy import stats
from dataclasses import dataclass, field
import json
import os


@dataclass
class ExperimentMetrics:
    """Container for all metrics across a single experiment run."""
    seed: int
    method: str  # 'baseline', 'weight_decay', 'ewc', 'functional_trust', 'kl_trust'
    experiment: str  # 'continual_cifar', 'transformer', 'rl_gridworld'

    # Per-step tracking
    steps: List[int] = field(default_factory=list)
    task_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)  # Or rewards for RL
    functional_drifts: List[float] = field(default_factory=list)
    param_norm_drifts: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    lambdas: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    cka_similarities: List[float] = field(default_factory=list)

    # Per-task tracking (for continual learning)
    task_accuracies: Dict[str, List[float]] = field(default_factory=dict)
    forgetting_scores: List[float] = field(default_factory=list)

    # RL-specific
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a single training step."""
        self.steps.append(step)
        for key, val in metrics.items():
            if hasattr(self, key + 's') and isinstance(getattr(self, key + 's'), list):
                getattr(self, key + 's').append(val)
            elif hasattr(self, key) and isinstance(getattr(self, key), list):
                getattr(self, key).append(val)

    def compute_forgetting(self, task_id: str, initial_acc: float, current_acc: float):
        """
        Forgetting score for a specific task.
        F_j = max_{t'<t} a_{t',j} - a_{t,j}
        where a_{t,j} is accuracy on task j at time t.
        """
        forgetting = max(0.0, initial_acc - current_acc)
        self.forgetting_scores.append(forgetting)
        return forgetting

    def save(self, path: str):
        """Save metrics to JSON.
        
        Produces both flat lists (for StatisticalAnalyzer.aggregate) and
        a 'steps' list-of-dicts (for visualization/statistical_analysis).
        """
        # Build per-step dicts for downstream consumers
        step_dicts = []
        for i, step_num in enumerate(self.steps):
            d = {'step': step_num}
            for attr, key in [
                ('task_losses', 'task_loss'), ('total_losses', 'total_loss'),
                ('accuracies', 'accuracy'), ('functional_drifts', 'functional_drift'),
                ('param_norm_drifts', 'param_norm_drift'), ('grad_norms', 'grad_norm'),
                ('lambdas', 'lambda'), ('epsilons', 'epsilon'),
                ('cka_similarities', 'cka_similarity'),
            ]:
                lst = getattr(self, attr)
                if i < len(lst):
                    d[key] = lst[i]
            step_dicts.append(d)

        data = {
            'seed': self.seed,
            'method': self.method,
            'experiment': self.experiment,
            'steps': step_dicts,
            'task_losses': self.task_losses,
            'total_losses': self.total_losses,
            'accuracies': self.accuracies,
            'functional_drifts': self.functional_drifts,
            'param_norm_drifts': self.param_norm_drifts,
            'grad_norms': self.grad_norms,
            'lambdas': self.lambdas,
            'epsilons': self.epsilons,
            'cka_similarities': self.cka_similarities,
            'task_accuracies': self.task_accuracies,
            'forgetting_scores': self.forgetting_scores,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentMetrics':
        with open(path, 'r') as f:
            data = json.load(f)
        obj = cls(seed=data['seed'], method=data['method'], experiment=data['experiment'])
        for key, val in data.items():
            if hasattr(obj, key):
                setattr(obj, key, val)
        return obj


class StatisticalAnalyzer:
    """
    Performs rigorous statistical analysis across multiple seeds.
    
    Implements:
        - Mean ± std aggregation
        - 95% confidence intervals
        - Welch's t-test (unequal variance)
        - Cohen's d effect size
    """

    @staticmethod
    def aggregate(runs: List[ExperimentMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across seeds. Returns mean, std, CI."""
        if not runs:
            return {}

        # Find common minimum length
        min_steps = min(len(r.steps) for r in runs)

        result = {}
        for attr in ['task_losses', 'accuracies', 'functional_drifts',
                      'param_norm_drifts', 'grad_norms']:
            values = []
            for r in runs:
                data = getattr(r, attr)[:min_steps]
                values.append(data)

            values_arr = np.array(values)  # (n_seeds, n_steps)
            mean = np.mean(values_arr, axis=0)
            std = np.std(values_arr, axis=0, ddof=1)
            n = len(runs)
            ci_95 = 1.96 * std / np.sqrt(n)

            result[attr] = {
                'mean': mean.tolist(),
                'std': std.tolist(),
                'ci_lower': (mean - ci_95).tolist(),
                'ci_upper': (mean + ci_95).tolist(),
                'n_seeds': n,
            }

        # Final metrics (last value)
        for attr in ['accuracies', 'functional_drifts']:
            finals = [getattr(r, attr)[-1] if getattr(r, attr) else 0.0 for r in runs]
            result[f'final_{attr}'] = {
                'mean': float(np.mean(finals)),
                'std': float(np.std(finals, ddof=1)),
                'values': finals,
            }

        # Forgetting
        all_forgetting = [np.mean(r.forgetting_scores) if r.forgetting_scores else 0.0 for r in runs]
        result['mean_forgetting'] = {
            'mean': float(np.mean(all_forgetting)),
            'std': float(np.std(all_forgetting, ddof=1)),
            'values': all_forgetting,
        }

        return result

    @staticmethod
    def welch_t_test(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, float]:
        """
        Welch's t-test for comparing two methods.
        Does not assume equal variance.
        
        Returns:
            t_statistic, p_value, cohens_d, significant (at α=0.05)
        """
        a = np.array(group_a)
        b = np.array(group_b)

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        cohens_d = (a.mean() - b.mean()) / max(pooled_std, 1e-10)

        return {
            't_stat': float(t_stat),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'significant': bool(p_val < 0.05),
            'mean_a': float(a.mean()),
            'mean_b': float(b.mean()),
            'std_a': float(a.std(ddof=1)),
            'std_b': float(b.std(ddof=1)),
        }

    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Compute confidence interval."""
        arr = np.array(data)
        n = len(arr)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return {
            'mean': float(mean),
            'ci_lower': float(mean - h),
            'ci_upper': float(mean + h),
            'std': float(arr.std(ddof=1)),
            'se': float(se),
        }
