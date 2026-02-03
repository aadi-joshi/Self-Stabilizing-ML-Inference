# Self-Stabilizing Inference System: Comprehensive Report

## Overview
This project implements and benchmarks a dual-signal self-stabilizing inference system for robust model selection under dynamic and adversarial conditions. The system is modular, extensible, and supports a variety of controllers and environmental challenges.

---

## Directory Structure (src/ only)
- `main.py`: Experiment orchestration, logging, and comparative evaluation
- `controller/dual_controller.py`: Main state machine controller (multi-objective, adaptive, oscillation-aware)
- `controller/baseline_controllers.py`: Baseline controllers (always-fast, always-robust, threshold-only, smoothing-only)
- `environment/degradation.py`: Environment simulation (bursty failures, drift, adversarial oscillation)
- `environment/injector.py`: Fault injection (latency, noise)
- `metrics/`: Reliability, latency, and smoothing modules
- `models/`: Fragile and robust model definitions
- `monitoring/telemetry.py`: Logging
- `visualization/plots.py`: Plotting utilities

---

## Iterations & Features
### Iteration 1: Basic Dual-Signal Control
- Reliability and latency measured and smoothed (EWMA)
- State machine controller: STABLE, DEGRADED, RECOVERING
- Model switching based on thresholds

### Iteration 2: Adaptive Smoothing
- Rolling variance computed for reliability/latency
- Smoother alpha increases during high variance
- All smoothing parameters configurable in YAML

### Iteration 3: Predictive Degradation Detection
- First-order derivative and volatility of smoothed reliability
- PREEMPTIVE_DEGRADED state triggered on persistent negative trend
- Logs predicted/actual degradation steps and lead time
- Plots mark prediction/actual events

### Iteration 4: Multi-Objective Controller
- Cost function: $J = \alpha (1 - \text{reliability}) + \beta \cdot \text{latency} + \gamma \cdot \text{switch penalty}$
- Controller selects model minimizing $J$ (short horizon)
- Switch penalty increases under oscillation
- All weights configurable

### Iteration 5: Oscillation Detection & Adaptive Dwell
- Sliding window tracks switch frequency
- Dwell time increases during oscillation, resets after stabilization
- Logs stabilization time after oscillatory events

### Iteration 6: Baseline Controllers & Comparative Evaluation
- Always-fast, always-robust, threshold-only, smoothing-only controllers
- All controllers run under identical conditions
- Comparative plots and summary tables generated

### Iteration 7: Advanced Environment & Robustness Metrics
- Bursty failures, gradual drift, adversarial oscillation attempts
- Measures: recovery time, oscillation count, stability duration
- All results and plots saved for each controller

---

## Key Modules & Logic
### Environment Simulation
- **Bursty Failures:** Periodic high-noise/latency bursts
- **Gradual Drift:** Slowly increasing noise after a configurable step
- **Adversarial Oscillation:** Sinusoidal noise/latency to induce controller oscillation

### Controller Logic
- **State Machine:** STABLE, DEGRADED, RECOVERING, PREEMPTIVE_DEGRADED
- **Multi-Objective Cost:** $J$ combines reliability, latency, and switching penalty
- **Oscillation Detection:** Adaptive dwell time, stabilization logging
- **Baselines:** Always-fast, always-robust, threshold-only, smoothing-only

### Metrics & Logging
- **Reliability:** Output variance under noise
- **Latency:** Model inference time
- **Smoothing:** EWMA, adaptive alpha
- **Telemetry:** All signals, states, and events logged per step

---

## Results & Plots
All results are saved under `results/metrics/iteration_X/` and plots under `plots/iteration_X/` for each iteration.

### Example: Iteration 7 Summary Table
| Controller      | Avg Reliability | Avg Latency | Oscillation Count | Recovery Time | Max Stability Duration |
|----------------|-----------------|-------------|-------------------|---------------|-----------------------|
| main           | ...             | ...         | ...               | ...           | ...                   |
| always_fast    | ...             | ...         | ...               | ...           | ...                   |
| always_robust  | ...             | ...         | ...               | ...           | ...                   |
| threshold_only | ...             | ...         | ...               | ...           | ...                   |
| smoothing_only | ...             | ...         | ...               | ...           | ...                   |

### Example Plots
- `reliability_comparison.png`: Smoothed reliability for all controllers
- `latency_comparison.png`: Smoothed latency for all controllers
- Plots mark predicted/actual degradation, oscillation, and recovery events

---

## Configuration (YAML)
All parameters (thresholds, smoothing, cost weights, environment) are fully configurable in the YAML file. Example:
```yaml
degradation:
  threshold: 0.65
  ewma_alpha: 0.2
  burst_period: 100
  burst_length: 10
  burst_noise: 0.3
  drift_start: 200
  drift_rate: 0.0005
  osc_start: 350
  osc_period: 20
  osc_amplitude: 0.12
control:
  alpha: 1.0
  beta: 1.0
  gamma: 0.1
  cooldown_steps: 30
```

---

## How to Reproduce
1. Install requirements (see requirements.txt)
2. Run `python src/main.py`
3. Inspect results in `results/metrics/iteration_X/` and plots in `plots/iteration_X/`
4. Adjust YAML config for further experiments

---

## Conclusion
This codebase provides a robust, extensible framework for evaluating self-stabilizing inference and control under a wide range of real-world and adversarial conditions. All logic, results, and plots are reproducible and fully documented.

---

*For further details, see the code and outputs in the src/ directory and results folders.*
