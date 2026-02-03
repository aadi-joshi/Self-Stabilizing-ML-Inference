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

# Controller (multi-objective)
alpha = config['control'].get('alpha', 1.0)
beta = config['control'].get('beta', 1.0)
gamma = config['control'].get('gamma', 0.1)
horizon = config['control'].get('horizon', 1)
controller = DualSignalController(
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    horizon=horizon,
    min_dwell_steps=config['control'].get('cooldown_steps', 0)
)


# Telemetry
logger = TelemetryLogger()

# Predictive degradation detection state
from collections import deque
deriv_window = config['degradation'].get('predictive_deriv_window', 5)
vol_window = config['degradation'].get('predictive_vol_window', 10)
neg_deriv_thresh = config['degradation'].get('predictive_neg_deriv_thresh', -0.002)
vol_thresh = config['degradation'].get('predictive_vol_thresh', 0.01)
neg_trend_steps = config['degradation'].get('predictive_neg_trend_steps', 3)

recent_smoothed = deque(maxlen=vol_window)
recent_derivs = deque(maxlen=deriv_window)
neg_trend_count = 0
predicted_degradation_step = None
actual_degradation_step = None
lead_time = None
preemptive_triggered = False

# Experiment loop
steps = 500
active_model = 'fast'
last_switch_step = -100
stability_state = StabilityState.STABLE


for step in range(steps):
    x = injector.inject(np.random.uniform(-1, 1, size=(2,)), step=step, env=env)
    noise = env.get_noise(step)

    import torch
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.tensor(x, dtype=torch.float32)
    else:
        x_tensor = x

    # Evaluate both models for cost function
    fast_reliability = reliability_metric.compute(fast_model, x, noise)
    fast_latency = latency_metric.measure(fast_model, x_tensor)
    robust_reliability = reliability_metric.compute(robust_model, x, noise)
    robust_latency = latency_metric.measure(robust_model, x_tensor)

    # Use current model for smoothing
    if active_model == 'fast':
        reliability = fast_reliability
        latency = fast_latency
    else:
        reliability = robust_reliability
        latency = robust_latency

    smoothed_reliability = reliability_smoother.update(reliability)
    smoothed_latency = latency_smoother.update(latency)

    # Predictive degradation detection (unchanged)
    recent_smoothed.append(smoothed_reliability)
    if len(recent_smoothed) > 1:
        deriv = recent_smoothed[-1] - recent_smoothed[-2]
        recent_derivs.append(deriv)
    else:
        deriv = 0.0

    rolling_vol = np.std(recent_smoothed) if len(recent_smoothed) >= 2 else 0.0
    mean_deriv = np.mean(recent_derivs) if len(recent_derivs) == deriv_window else 0.0

    if mean_deriv < neg_deriv_thresh and rolling_vol > vol_thresh:
        neg_trend_count += 1
    else:
        neg_trend_count = 0

    if neg_trend_count >= neg_trend_steps and not preemptive_triggered:
        preemptive_triggered = True
        predicted_degradation_step = step
        print(f"[STEP {step}] PREEMPTIVE_DEGRADED triggered (prediction)")
        stability_state = StabilityState.PREEMPTIVE_DEGRADED

    if stability_state != StabilityState.DEGRADED and 'new_state' in locals() and new_state == StabilityState.DEGRADED:
        actual_degradation_step = step
        if predicted_degradation_step is not None:
            lead_time = predicted_degradation_step - actual_degradation_step
        print(f"[STEP {step}] Actual DEGRADED state detected")

    # Pass both model predictions to controller
    action, new_state = controller.decide(
        smoothed_reliability, smoothed_latency, stability_state, step, last_switch_step, active_model,
        fast_pred=(fast_reliability, fast_latency),
        robust_pred=(robust_reliability, robust_latency)
    )

    if action:
        active_model = action
        last_switch_step = step
        print(f"[STEP {step}] Controller state transition: {stability_state.name} → {new_state.name}")
        print(f"[STEP {step}] Switching to {active_model.upper()} model due to state {new_state.name}")
    elif new_state != stability_state:
        print(f"[STEP {step}] Controller state transition: {stability_state.name} → {new_state.name}")
    stability_state = new_state

    logger.log({
        'step': step,
        'reliability': reliability,
        'smoothed_reliability': smoothed_reliability,
        'latency': latency,
        'smoothed_latency': smoothed_latency,
        'active_model': active_model,
        'controller_state': stability_state.name,
        'deriv': deriv,
        'rolling_vol': rolling_vol,
        'mean_deriv': mean_deriv,
        'predicted_degradation_step': predicted_degradation_step,
        'actual_degradation_step': actual_degradation_step,
        'lead_time': lead_time,
        'fast_J': controller.alpha * (1 - fast_reliability) + controller.beta * fast_latency,
        'robust_J': controller.alpha * (1 - robust_reliability) + controller.beta * robust_latency
    })

# Artifact storage
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results/logs/iteration_4/{run_id}'
metrics_dir = f'results/metrics/iteration_4/{run_id}'
plots_dir = f'plots/iteration_4/{run_id}'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Save logs and metrics
logger.df.to_csv(os.path.join(results_dir, 'telemetry.csv'), index=False)
logger.df[['reliability','smoothed_reliability','latency','smoothed_latency','deriv','rolling_vol','mean_deriv','predicted_degradation_step','actual_degradation_step','lead_time','fast_J','robust_J']].to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)

# Visualization
plot_all(logger.df, plots_dir)

print('Experiment finished.')
