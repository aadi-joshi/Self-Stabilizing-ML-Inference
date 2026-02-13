# ============================================================================
# Experiment C: RL Gridworld with Policy Drift Constraint
# Policy learning with and without functional constraint
# ============================================================================

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rl_agent import GridWorld, PolicyNetwork, RolloutBuffer
from metrics.functional_drift import FunctionalDrift
from metrics.constrained_optimizer import EpsilonScheduler
from metrics.experiment_metrics import ExperimentMetrics
from utils.common import set_seed, AverageMeter


def run_rl_experiment(
    method: str,
    config: Dict,
    seed: int,
    device: torch.device,
    save_dir: str,
) -> ExperimentMetrics:
    """
    Run RL gridworld experiment with optional functional drift constraint.
    
    Methods:
        - 'baseline': Standard REINFORCE
        - 'weight_decay': REINFORCE + weight decay
        - 'functional_trust': REINFORCE + functional drift constraint
        - 'kl_trust': REINFORCE + KL trust region
    """
    set_seed(seed)
    exp_cfg = config['experiment_c']

    grid_size = exp_cfg.get('grid_size', 8)
    env = GridWorld(size=grid_size, seed=seed)

    model_cfg = exp_cfg.get('model', {})
    policy = PolicyNetwork(
        n_states=env.n_states,
        n_actions=env.n_actions,
        hidden_dim=model_cfg.get('hidden_dim', 64),
        num_layers=model_cfg.get('num_layers', 2),
    ).to(device)

    lr = exp_cfg.get('lr', 0.001)
    gamma = exp_cfg.get('gamma', 0.99)
    n_episodes = exp_cfg.get('episodes', 2000)

    if method == "weight_decay":
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Reference data for drift measurement: sample of state space
    ref_states = torch.eye(env.n_states, device=device)  # All possible states

    # Setup drift measurement
    drift_module = FunctionalDrift(
        reference_model=policy, reference_data=ref_states,
        norm_type='kl' if method == 'kl_trust' else 'l2', device=device
    )

    # For KL trust region: keep reference policy
    if method == 'kl_trust':
        ref_policy = copy.deepcopy(policy).to(device)
        ref_policy.eval()
        kl_coeff = exp_cfg.get('kl_coeff', 0.01)

    # For functional trust
    lambda_val = exp_cfg.get('drift_lambda', 0.5)
    lambda_lr = config['drift'].get('lambda_lr', 0.01)
    eps_scheduler = EpsilonScheduler(
        schedule_type=config['epsilon_scheduler'].get('type', 'fixed'),
        epsilon_init=exp_cfg.get('drift_epsilon', 0.5),
        epsilon_min=config['drift'].get('epsilon_min', 0.01),
        total_steps=n_episodes,
    ) if method == 'functional_trust' else None

    metrics = ExperimentMetrics(seed=seed, method=method, experiment='rl_gridworld')
    initial_policy = copy.deepcopy(policy)

    # --- Training Loop ---
    update_interval = 10  # Update policy every N episodes
    buffer = RolloutBuffer()
    episode_reward_meter = AverageMeter()

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action, log_prob, value = policy.get_action(state, device)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, log_prob, reward, value, done)
            state = next_state
            ep_reward += reward
            ep_length += 1

        metrics.episode_rewards.append(ep_reward)
        metrics.episode_lengths.append(ep_length)
        episode_reward_meter.update(ep_reward)

        # Policy update every update_interval episodes
        if (episode + 1) % update_interval == 0 and len(buffer.rewards) > 0:
            returns = buffer.compute_returns(gamma).to(device)
            log_probs = torch.stack(buffer.log_probs).to(device)
            values = torch.cat(buffer.values).to(device).squeeze()

            # Advantage
            advantage = returns - values.detach()

            # Policy gradient loss
            policy_loss = -(log_probs * advantage.detach()).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            task_loss = policy_loss + 0.5 * value_loss

            # Method-specific regularization
            if method == "functional_trust":
                # Compute differentiable drift
                drift_loss = drift_module.compute_differentiable(policy)
                total_loss = task_loss + lambda_val * drift_loss

                # Dual update
                epsilon = eps_scheduler.get_epsilon(episode) if eps_scheduler else 0.5
                drift_val = drift_loss.item()
                violation = drift_val - epsilon
                lambda_val = max(0.0, lambda_val + lambda_lr * violation)

            elif method == "kl_trust":
                # KL divergence between current and reference policy
                states_t = torch.FloatTensor(np.array(buffer.states)).to(device)
                curr_probs, _ = policy(states_t)
                with torch.no_grad():
                    ref_probs, _ = ref_policy(states_t)
                kl = F.kl_div(
                    curr_probs.log(), ref_probs, reduction='batchmean'
                )
                total_loss = task_loss + kl_coeff * kl
            else:
                total_loss = task_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # Update KL reference periodically
            if method == 'kl_trust' and (episode + 1) % 100 == 0:
                ref_policy = copy.deepcopy(policy).to(device)
                ref_policy.eval()

            buffer.clear()

        # Log metrics
        if (episode + 1) % config['metrics'].get('log_interval', 10) == 0:
            drift_info = drift_module.compute(policy)

            log_data = {
                'accuracy': ep_reward,  # Using reward as "accuracy" for RL
                'task_loss': ep_reward,
                'functional_drift': drift_info['drift'],
                'param_norm_drift': sum(
                    (p1 - p2).norm().item()
                    for p1, p2 in zip(policy.parameters(), initial_policy.parameters())
                ),
            }
            if method == 'functional_trust':
                log_data['lambda'] = lambda_val
                log_data['epsilon'] = eps_scheduler.get_epsilon(episode) if eps_scheduler else 0.5

            metrics.log_step(episode, log_data)

        if (episode + 1) % 200 == 0:
            print(f"    Episode {episode + 1}: avg_reward={episode_reward_meter.avg:.3f}, "
                  f"drift={drift_info['drift'] if drift_info else 0:.4f}")

    metrics_path = os.path.join(save_dir, f"{method}_seed{seed}_metrics.json")
    metrics.save(metrics_path)
    return metrics
