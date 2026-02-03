# Self-Stabilizing Inference System: Comprehensive Report

## Overview
This project implements and benchmarks a dual-signal self-stabilizing inference system for robust model selection under dynamic and adversarial conditions. The system is modular, extensible, and supports a variety of controllers and environmental challenges.


## Directory Structure (src/ only)


## Iterations & Features
### Iteration 1: Basic Dual-Signal Control

### Iteration 2: Adaptive Smoothing

### Iteration 3: Predictive Degradation Detection

### Iteration 4: Multi-Objective Controller

### Iteration 5: Oscillation Detection & Adaptive Dwell

### Iteration 6: Baseline Controllers & Comparative Evaluation

### Iteration 7: Advanced Environment & Robustness Metrics


## Key Modules & Logic
### Environment Simulation

### Controller Logic

### Metrics & Logging


## Results & Plots
All results are saved under `results/metrics/iteration_X/` and plots under `plots/iteration_X/` for each iteration.


### Iteration 7: Controller Comparison Summary

```
controller,avg_reliability,avg_latency,oscillation_count,recovery_time,max_stability_duration
always_fast,0.9283,4.22e-05,1,1.0,499
always_robust,0.9589,2.22e-05,1,,500
threshold_only,0.9290,2.40e-05,16,1.0,499
smoothing_only,0.9280,2.41e-05,2,1.0,499
main,0.9567,3.05e-05,5,1.0,499
```

### Example Metrics (main controller, first 10 steps)

```
step,reliability,smoothed_reliability,latency,smoothed_latency,active_model,controller_state,deriv,fast_J,robust_J
0,0.6457,0.6457,6.94e-05,6.94e-05,robust,StabilityState.DEGRADED,0.0,0.3544,0.2038
1,0.7702,0.6706,2.46e-05,6.05e-05,fast,StabilityState.STABLE,0.0249,0.0831,0.2298
2,0.8323,0.7029,2.51e-05,5.34e-05,fast,StabilityState.STABLE,0.0323,0.1677,0.1940
3,0.8131,0.7250,2.88e-05,4.85e-05,robust,StabilityState.DEGRADED,0.0220,0.1869,0.0854
4,0.9086,0.7617,2.23e-05,4.32e-05,robust,StabilityState.DEGRADED,0.0367,0.1062,0.0915
5,0.9001,0.7894,2.25e-05,3.91e-05,robust,StabilityState.DEGRADED,0.0277,0.0656,0.0999
6,0.8264,0.7968,2.40e-05,3.61e-05,robust,StabilityState.DEGRADED,0.0074,0.5726,0.1736
7,0.8300,0.8034,2.29e-05,3.34e-05,robust,StabilityState.DEGRADED,0.0066,0.1050,0.1701
8,0.8861,0.8200,2.27e-05,3.13e-05,robust,StabilityState.DEGRADED,0.0165,0.1378,0.1139
9,0.8560,0.8272,2.23e-05,2.95e-05,robust,StabilityState.DEGRADED,0.0072,0.3531,0.1440
```

- **Latency Comparison:** ![Latency Comparison](plots/iteration_7/20260203_213048/latency_comparison.png)

> For full metrics, see the CSV files in `results/metrics/iteration_7/20260203_213048/`.

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
