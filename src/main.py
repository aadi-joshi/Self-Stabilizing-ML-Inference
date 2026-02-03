# Main entrypoint for dual-signal self-stabilizing inference system
# Loads config, orchestrates experiment, logging, and visualization

import os
import yaml
import numpy as np
import torch
import random
from datetime import datetime

from models.fragile_model import FragileModel
from models.robust_model import RobustModel
from environment.data import generate_data
from environment.degradation import EnvironmentDegradation
from environment.injector import FaultInjector
from metrics.reliability import ReliabilityMetric
from metrics.latency import LatencyMetric
from metrics.smoothing import ExponentialSmoother
from controller.dual_controller import DualSignalController, StabilityState
from controller.baseline_controllers import AlwaysFastController, AlwaysRobustController, ThresholdOnlyController, SmoothingOnlyController
from monitoring.telemetry import TelemetryLogger
from visualization.plots import plot_all

# Load config
import os
config_path = os.path.join(os.path.dirname(__file__), '../self_stabilizing_inference/config/config.yaml')
config_path = os.path.abspath(config_path)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

random_seed = config['random_seed']
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

# Data
X_train, y_train = generate_data(n=2000)

# Models
fast_model = FragileModel(input_dim=2, num_classes=2)
robust_model = RobustModel(input_dim=2, num_classes=2)
# TODO: Add training logic if needed

# Environment
env = EnvironmentDegradation(config['degradation'])
injector = FaultInjector(
    latency_prob=config['faults']['latency_spike_prob'],
    noise_prob=config['faults']['noise_prob']
)

# Metrics
reliability_metric = ReliabilityMetric()
latency_metric = LatencyMetric()
reliability_smoother = ExponentialSmoother(alpha=config['degradation']['ewma_alpha'])
latency_smoother = ExponentialSmoother(alpha=config['degradation'].get('latency_ewma_alpha', 0.2))


# Controller configs
alpha = config['control'].get('alpha', 1.0)
beta = config['control'].get('beta', 1.0)
gamma = config['control'].get('gamma', 0.1)
horizon = config['control'].get('horizon', 1)
main_controller = DualSignalController(
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    horizon=horizon,
    min_dwell_steps=config['control'].get('cooldown_steps', 0)
)
baseline_controllers = {
    'always_fast': AlwaysFastController(),
    'always_robust': AlwaysRobustController(),
    'threshold_only': ThresholdOnlyController(
        reliability_threshold=config['degradation']['threshold'],
        latency_threshold=config['degradation'].get('latency_threshold', 0.1)),
    'smoothing_only': SmoothingOnlyController(
        reliability_threshold=config['degradation']['threshold'],
        latency_threshold=config['degradation'].get('latency_threshold', 0.1)),
    'main': main_controller
}



# Run all controllers under identical conditions
from collections import deque
import pandas as pd
steps = 500
controller_results = {}
for ctrl_name, controller in baseline_controllers.items():
    # Reset seeds for identical conditions
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Reset environment and metrics
    env = EnvironmentDegradation(config['degradation'])
    injector = FaultInjector(
        latency_prob=config['faults']['latency_spike_prob'],
        noise_prob=config['faults']['noise_prob']
    )
    reliability_metric = ReliabilityMetric()
    latency_metric = LatencyMetric()
    reliability_smoother = ExponentialSmoother(alpha=config['degradation']['ewma_alpha'])
    latency_smoother = ExponentialSmoother(alpha=config['degradation'].get('latency_ewma_alpha', 0.2))

    logger = TelemetryLogger()
    deriv_window = config['degradation'].get('predictive_deriv_window', 5)
    vol_window = config['degradation'].get('predictive_vol_window', 10)
    recent_smoothed = deque(maxlen=vol_window)
    recent_derivs = deque(maxlen=deriv_window)
    neg_trend_count = 0
    predicted_degradation_step = None
    actual_degradation_step = None
    lead_time = None
    preemptive_triggered = False

    active_model = 'fast'
    last_switch_step = -100
    stability_state = StabilityState.STABLE if hasattr(controller, 'decide') and hasattr(controller, 'osc_window') else 'fast'

    for step in range(steps):
        x = injector.inject(np.random.uniform(-1, 1, size=(2,)), step=step, env=env)
        noise = env.get_noise(step)
        if not isinstance(x, torch.Tensor):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
        fast_reliability = reliability_metric.compute(fast_model, x, noise)
        fast_latency = latency_metric.measure(fast_model, x_tensor)
        robust_reliability = reliability_metric.compute(robust_model, x, noise)
        robust_latency = latency_metric.measure(robust_model, x_tensor)
        if ctrl_name == 'always_fast':
            reliability = fast_reliability
            latency = fast_latency
        elif ctrl_name == 'always_robust':
            reliability = robust_reliability
            latency = robust_latency
        elif ctrl_name == 'threshold_only':
            reliability = fast_reliability if active_model == 'fast' else robust_reliability
            latency = fast_latency if active_model == 'fast' else robust_latency
        elif ctrl_name == 'smoothing_only':
            reliability = fast_reliability if active_model == 'fast' else robust_reliability
            latency = fast_latency if active_model == 'fast' else robust_latency
        else:
            reliability = fast_reliability if active_model == 'fast' else robust_reliability
            latency = fast_latency if active_model == 'fast' else robust_latency

        smoothed_reliability = reliability_smoother.update(reliability)
        smoothed_latency = latency_smoother.update(latency)

        recent_smoothed.append(smoothed_reliability)
        if len(recent_smoothed) > 1:
            deriv = recent_smoothed[-1] - recent_smoothed[-2]
            recent_derivs.append(deriv)
        else:
            deriv = 0.0

        # Controller decision
        if ctrl_name == 'always_fast':
            action, new_state, oscillating, stabilization_time = controller.decide()
        elif ctrl_name == 'always_robust':
            action, new_state, oscillating, stabilization_time = controller.decide()
        elif ctrl_name == 'threshold_only':
            action, new_state, oscillating, stabilization_time = controller.decide(reliability, latency)
        elif ctrl_name == 'smoothing_only':
            action, new_state, oscillating, stabilization_time = controller.decide(smoothed_reliability, smoothed_latency)
        else:
            action, new_state, oscillating, stabilization_time = controller.decide(
                smoothed_reliability, smoothed_latency, stability_state, step, last_switch_step, active_model,
                fast_pred=(fast_reliability, fast_latency),
                robust_pred=(robust_reliability, robust_latency)
            )

        if action:
            active_model = action
            last_switch_step = step
        elif new_state != stability_state:
            pass
        stability_state = new_state

        logger.log({
            'step': step,
            'reliability': reliability,
            'smoothed_reliability': smoothed_reliability,
            'latency': latency,
            'smoothed_latency': smoothed_latency,
            'active_model': active_model,
            'controller_state': str(stability_state),
            'deriv': deriv,
            'fast_J': alpha * (1 - fast_reliability) + beta * fast_latency,
            'robust_J': alpha * (1 - robust_reliability) + beta * robust_latency
        })
    controller_results[ctrl_name] = logger.df.copy()

# Save and plot results

# Save and plot results for iteration_7
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
metrics_dir = f'results/metrics/iteration_7/{run_id}'
plots_dir = f'plots/iteration_7/{run_id}'
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
for ctrl_name, df in controller_results.items():
    df.to_csv(os.path.join(metrics_dir, f'{ctrl_name}_metrics.csv'), index=False)

# Comparative plots
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for ctrl_name, df in controller_results.items():
    plt.plot(df['smoothed_reliability'], label=f'{ctrl_name} reliability')
plt.xlabel('Step')
plt.ylabel('Smoothed Reliability')
plt.title('Smoothed Reliability Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'reliability_comparison.png'))
plt.close()

plt.figure(figsize=(12, 6))
for ctrl_name, df in controller_results.items():
    plt.plot(df['smoothed_latency'], label=f'{ctrl_name} latency')
plt.xlabel('Step')
plt.ylabel('Smoothed Latency')
plt.title('Smoothed Latency Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'latency_comparison.png'))
plt.close()

# Recovery time, oscillation count, stability duration
summary = []
for ctrl_name, df in controller_results.items():
    avg_reliability = df['smoothed_reliability'].mean()
    avg_latency = df['smoothed_latency'].mean()
    switches = (df['active_model'] != df['active_model'].shift()).sum()
    # Recovery time: steps from first drop below threshold to recovery above threshold
    threshold = config['degradation']['threshold']
    below = df['smoothed_reliability'] < threshold
    recovery_time = None
    if below.any():
        drop_idx = below.idxmax()
        recovered = (df['smoothed_reliability'][drop_idx:] > threshold)
        if recovered.any():
            recovery_time = recovered.idxmax() - drop_idx
    # Oscillation count: number of switches
    oscillation_count = switches
    # Stability duration: longest consecutive period above threshold
    stable = df['smoothed_reliability'] > threshold
    max_stable = (stable.groupby((stable != stable.shift()).cumsum()).cumsum() * stable).max()
    summary.append({'controller': ctrl_name, 'avg_reliability': avg_reliability, 'avg_latency': avg_latency, 'oscillation_count': oscillation_count, 'recovery_time': recovery_time, 'max_stability_duration': max_stable})
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(metrics_dir, 'controller_comparison_summary.csv'), index=False)
print(summary_df)
