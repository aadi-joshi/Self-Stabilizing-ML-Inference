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

# Controller
controller = DualSignalController(
    reliability_threshold=config['degradation']['threshold'],
    latency_threshold=config['degradation'].get('latency_threshold', 0.1),
    min_dwell_steps=config['control']['cooldown_steps']
)

# Telemetry
logger = TelemetryLogger()

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

    if active_model == 'fast':
        model = fast_model
    else:
        model = robust_model

    reliability = reliability_metric.compute(model, x, noise)
    latency = latency_metric.measure(model, x_tensor)

    smoothed_reliability = reliability_smoother.update(reliability)
    smoothed_latency = latency_smoother.update(latency)

    action, new_state = controller.decide(
        smoothed_reliability, smoothed_latency, stability_state, step, last_switch_step, active_model
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
        'controller_state': stability_state.name
    })

# Artifact storage
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results/logs/{run_id}'
metrics_dir = f'results/metrics/{run_id}'
plots_dir = f'plots/iteration_1/{run_id}'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Save logs and metrics
logger.df.to_csv(os.path.join(results_dir, 'telemetry.csv'), index=False)
logger.df[['reliability','smoothed_reliability','latency','smoothed_latency']].to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)

# Visualization
plot_all(logger.df, plots_dir)

print('Experiment finished.')
