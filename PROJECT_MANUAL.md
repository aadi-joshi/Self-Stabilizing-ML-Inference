# Self-Stabilizing Machine Learning Inference System — Complete Project Manual

> **Author:** Kavya Bhand  
> **License:** MIT License © 2026  
> **Last Updated:** 13 February 2026  
> **Version:** 1.0  

---

> ⚠️ **MAINTENANCE INSTRUCTION — KEEP THIS DOCUMENT UPDATED**  
> Every time any change is made to the codebase — new modules, modified logic, new experiments, updated configurations, additional visualizations, or refactored architecture — this document **must** be updated to reflect those changes. Treat this manual as the living, canonical source of truth for the entire project. When updating:  
> 1. Update the "Last Updated" date and increment the version at the top.  
> 2. Add a new entry in the [Change Log](#22-change-log) section at the end.  
> 3. Modify every affected section (architecture, module docs, math, results, etc.).  
> 4. If new experiments are run, add their results under [Section 14: Experimental Results](#14-experimental-results-and-analysis).  
> 5. If new plots are generated, reference them under [Section 15: Visualizations](#15-visualizations-and-plot-inventory).  
> 6. If the directory structure changes, update [Section 4: Directory Structure](#4-complete-directory-structure).  

---

## Table of Contents

1.  [Executive Summary](#1-executive-summary)
2.  [Problem Statement & Motivation](#2-problem-statement--motivation)
3.  [System Overview & Architecture](#3-system-overview--architecture)
4.  [Complete Directory Structure](#4-complete-directory-structure)
5.  [Module-by-Module Documentation](#5-module-by-module-documentation)
    - 5.1 [Data Generation](#51-data-generation)
    - 5.2 [Models](#52-models)
    - 5.3 [Environment & Degradation](#53-environment--degradation)
    - 5.4 [Fault Injection](#54-fault-injection)
    - 5.5 [Metrics](#55-metrics)
    - 5.6 [Smoothing](#56-smoothing)
    - 5.7 [Controllers](#57-controllers)
    - 5.8 [Monitoring & Telemetry](#58-monitoring--telemetry)
    - 5.9 [Visualization](#59-visualization)
    - 5.10 [Stability Metrics](#510-stability-metrics)
6.  [Mathematical Foundations](#6-mathematical-foundations)
7.  [Control Theory & State Machine](#7-control-theory--state-machine)
8.  [Multi-Objective Optimization](#8-multi-objective-optimization)
9.  [Reinforcement Learning Controller](#9-reinforcement-learning-controller)
10. [Predictive Degradation Detection](#10-predictive-degradation-detection)
11. [Configuration Reference](#11-configuration-reference)
12. [Data Flow — End-to-End Pipeline](#12-data-flow--end-to-end-pipeline)
13. [Iterative Development History](#13-iterative-development-history)
14. [Experimental Results and Analysis](#14-experimental-results-and-analysis)
15. [Visualizations and Plot Inventory](#15-visualizations-and-plot-inventory)
16. [Conclusions & Key Inferences](#16-conclusions--key-inferences)
17. [Dual Codebase Explanation](#17-dual-codebase-explanation)
18. [How to Reproduce](#18-how-to-reproduce)
19. [Extending the System](#19-extending-the-system)
20. [Known Limitations & Future Work](#20-known-limitations--future-work)
21. [Glossary](#21-glossary)
22. [Change Log](#22-change-log)

---

## 1. Executive Summary

This project implements a **self-stabilizing machine learning inference system** that dynamically selects between a fast (low-latency, fragile) model and a robust (higher-latency, resilient) model at inference time, based on real-time environmental conditions. The system continuously monitors two primary signals — **reliability** and **latency** — applies exponential smoothing, and uses a multi-objective stateful controller to make switching decisions. The goal is to maintain inference quality above a configurable threshold while minimizing latency and avoiding oscillatory model-switching behavior.

The project includes:

- A **dual-signal control framework** with hysteresis, dwell time, and oscillation detection.
- **Six different controllers** for comparative evaluation (Always Fast, Always Robust, Threshold-Only, Smoothing-Only, Dual-Signal Main, and Learning/RL Controller).
- **Two environment modes**: a structured, deterministic degradation pattern and an unseen random degradation pattern for robustness validation.
- **Formal stability metrics**: stability horizon, oscillation bound, and recovery time distribution.
- **Predictive degradation detection** via derivative analysis and rolling volatility.
- An **online Q-learning controller** that learns model-selection policies from experience.
- Comprehensive **telemetry logging**, **CSV export**, and **matplotlib visualizations** across 8+ experimental iterations.

---

## 2. Problem Statement & Motivation

### 2.1 The Problem

When deploying ML models in production environments, inference quality can degrade unpredictably due to:

- **Environmental noise**: sensor drift, data corruption, adversarial perturbations.
- **Latency spikes**: resource contention, network delays, I/O bottlenecks.
- **Bursty failures**: transient hardware/software faults.
- **Gradual degradation**: model staleness, concept drift.
- **Adversarial oscillations**: periodic perturbations designed to destabilize switching logic.

A single static model cannot handle all of these conditions. Using only a robust model wastes resources during stable periods. Using only a fast model risks catastrophic failure during degradations.

### 2.2 The Goal

Design a **closed-loop control system** that:

1. Continuously monitors inference quality (reliability and latency).
2. Smooths raw signals to avoid reacting to transient noise.
3. Switches to a more robust model when degradation is detected.
4. Switches back to a faster model when stability returns.
5. Avoids oscillatory switching (chattering) through hysteresis and dwell time.
6. Quantifies system stability using formal metrics.
7. Generalizes to unseen, random degradation patterns.

### 2.3 Why This Matters

This directly addresses real-world MLOps challenges:

- **Edge inference** where compute and network are unreliable.
- **Safety-critical systems** (medical, autonomous vehicles) requiring guaranteed reliability.
- **Cost-sensitive deployments** where using the cheapest acceptable model saves resources.
- **Adaptive serving** in cloud platforms with variable load.

---

## 3. System Overview & Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-STABILIZING INFERENCE SYSTEM             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │   Data    │───▶│  Fault    │───▶│  Model   │───▶│ Metrics  │ │
│  │Generator  │    │ Injector  │    │ Inference│    │ Compute  │ │
│  └──────────┘    └───────────┘    └──────────┘    └────┬─────┘ │
│                        ▲                               │       │
│                        │                               ▼       │
│  ┌──────────┐    ┌─────┴─────┐    ┌──────────┐  ┌──────────┐  │
│  │Environment│   │Controller │◀───│ Smoother │◀─│Reliability│  │
│  │Degradation│   │  (State   │    │  (EWMA)  │  │& Latency │  │
│  └──────────┘    │  Machine) │    └──────────┘  └──────────┘  │
│                  └─────┬─────┘                                 │
│                        │                                       │
│                        ▼                                       │
│                  ┌──────────┐    ┌──────────┐                  │
│                  │ Telemetry│───▶│  Plots   │                  │
│                  │  Logger  │    │& Reports │                  │
│                  └──────────┘    └──────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **Data Generator** | Generates 2D synthetic classification data | `src/environment/data.py` |
| **Models** | Fast (fragile) and Robust model architectures | `src/models/fragile_model.py`, `src/models/robust_model.py` |
| **Environment** | Simulates degradation patterns (noise, latency) | `src/environment/degradation.py`, `src/environment/random_degradation.py` |
| **Fault Injector** | Probabilistic noise and latency injection | `src/environment/injector.py` |
| **Reliability Metric** | Variance-based reliability scoring | `src/metrics/reliability.py` |
| **Latency Metric** | Wall-clock inference timing | `src/metrics/latency.py` |
| **Smoother** | Exponential weighted moving average | `src/metrics/smoothing.py` |
| **Controllers** | Model selection decision logic | `src/controller/dual_controller.py`, `src/controller/baseline_controllers.py`, `src/controller/learning_controller.py` |
| **Telemetry** | Per-step metric logging to DataFrame | `src/monitoring/telemetry.py` |
| **Visualization** | Plot generation for all metrics | `src/visualization/plots.py` |
| **Stability Metrics** | Formal stability analysis | `src/metrics/stability.py` |

### 3.3 Signal Flow Summary

```
Input x → FaultInjector(x) → Model(x') → [Reliability, Latency]
       → Smoother(Reliability) → Smoother(Latency)
       → Controller.decide(smoothed_r, smoothed_l, state, ...)
       → action ∈ {fast, robust, hold}
       → Update active_model, state
       → TelemetryLogger.log(...)
```

---

## 4. Complete Directory Structure

```
Self-Stabilizing-ML-Inference-System/
├── LICENSE                                    # MIT License
├── RELIABILITY_SMOOTHING_SUMMARY.md           # Summary doc: smoothing & thresholds
├── PROJECT_MANUAL.md                          # THIS DOCUMENT
│
├── self_stabilizing_inference/                # LEGACY / Prototype codebase (Keras/TF)
│   ├── main.py                                #   Standalone prototype control loop
│   ├── config/
│   │   └── config.yaml                        #   Shared YAML configuration
│   ├── control/
│   │   └── controller.py                      #   Simple cooldown-based controller
│   ├── data/
│   │   └── inputs.npy                         #   Precomputed input data
│   ├── detection/
│   │   └── degradation.py                     #   EWMA-based degradation detector
│   ├── experiment/
│   │   ├── run_experiment.py                  #   Experiment runner
│   │   └── system.py                          #   System orchestrator
│   ├── faults/
│   │   └── injector.py                        #   Time.sleep-based fault injector
│   ├── inference/
│   │   └── engine.py                          #   Inference engine (entropy, confidence)
│   ├── models/
│   │   ├── fragile_model.py                   #   PyTorch shallow model
│   │   └── robust_model.py                    #   PyTorch deeper model (Tanh)
│   ├── monitoring/
│   │   └── telemetry.py                       #   Telemetry logger
│   ├── reliability/
│   │   └── scoring.py                         #   Keras-based reliability scorer
│   ├── utils/
│   │   └── seed.py                            #   (Empty placeholder)
│   └── visualization/
│       └── plots.py                           #   Basic reliability plot
│
├── src/                                       # PRIMARY / Advanced codebase (PyTorch)
│   ├── main.py                                #   Main orchestrator: all controllers, envs
│   ├── CONTROL_LOGIC_AND_FLOW.md              #   Control flow documentation
│   ├── EXPERIMENT_REPORT.md                   #   Experiment report & results
│   │
│   ├── controller/
│   │   ├── dual_controller.py                 #   DualSignalController + StabilityState
│   │   ├── baseline_controllers.py            #   Always-Fast/Robust, Threshold, Smoothing
│   │   └── learning_controller.py             #   Q-Learning RL controller
│   │
│   ├── environment/
│   │   ├── data.py                            #   generate_data(): XOR-like 2D data
│   │   ├── degradation_interface.py           #   ABC: DegradationProcess interface
│   │   ├── degradation.py                     #   EnvironmentDegradation: structured patterns
│   │   ├── random_degradation.py              #   RandomDegradation: unseen random patterns
│   │   └── injector.py                        #   FaultInjector: probabilistic noise/latency
│   │
│   ├── metrics/
│   │   ├── reliability.py                     #   ReliabilityMetric: variance-based
│   │   ├── latency.py                         #   LatencyMetric: wall-clock timing
│   │   ├── smoothing.py                       #   ExponentialSmoother (EWMA)
│   │   └── stability.py                       #   Formal stability metrics
│   │
│   ├── models/
│   │   ├── fragile_model.py                   #   FragileModel: 2→128→2 (ReLU)
│   │   └── robust_model.py                    #   RobustModel: 2→64→2 (Tanh)
│   │
│   ├── monitoring/
│   │   └── telemetry.py                       #   TelemetryLogger: DataFrame-based
│   │
│   ├── visualization/
│   │   └── plots.py                           #   plot_all(): reliability, latency, state
│   │
│   ├── plots/                                 #   Generated plot images
│   │   ├── iteration_1/                       #   Basic dual-signal plots
│   │   ├── iteration_4/                       #   Multi-objective cost plots
│   │   ├── iteration_7/                       #   Controller comparison plots
│   │   └── iteration_8/                       #   Full comparison + stability + learning
│   │
│   └── results/                               #   Generated result CSVs
│       ├── logs/                              #   Telemetry logs per run
│       └── metrics/                           #   Per-step and summary CSVs
```

---

## 5. Module-by-Module Documentation

### 5.1 Data Generation

**File:** `src/environment/data.py`

```python
def generate_data(n=2000):
    X = np.random.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y
```

**Purpose:** Generates a 2D binary classification dataset where the label is determined by the sign of the product $x_1 \cdot x_2$ (an XOR-like decision boundary).

**Mathematics:**

$$
y = \begin{cases} 1 & \text{if } x_1 \cdot x_2 > 0 \\ 0 & \text{otherwise} \end{cases}
$$

This produces a non-linearly separable dataset requiring at least one hidden layer. The decision boundary consists of the two axes $x_1 = 0$ and $x_2 = 0$, forming four quadrants where opposite quadrants share the same class.

**Parameters:**
- `n` (int, default 2000): Number of data points.

**Returns:** Tuple `(X, y)` where `X` has shape `(n, 2)` and `y` has shape `(n,)`.

---

### 5.2 Models

#### 5.2.1 FragileModel (Fast Model)

**File:** `src/models/fragile_model.py`

```python
class FragileModel(nn.Module):
    net = nn.Sequential(
        nn.Linear(2, 128),   # Input → 128 hidden (ReLU)
        nn.ReLU(),
        nn.Linear(128, 2)    # 128 → 2 output classes
    )
```

**Architecture:** A single-hidden-layer feedforward neural network with 128 neurons and ReLU activation.

**Properties:**
- **Parameters:** $(2 \times 128 + 128) + (128 \times 2 + 2) = 386 + 258 = 642$ total parameters.
- **Speed:** Low latency due to small network depth.
- **Fragility:** Shallow architecture is more sensitive to input perturbations — predictions vary significantly with small noise, leading to higher output variance under degradation.
- **Activation:** ReLU is unbounded, making gradients and outputs more sensitive to adversarial noise.

#### 5.2.2 RobustModel

**File:** `src/models/robust_model.py`

```python
class RobustModel(nn.Module):
    net = nn.Sequential(
        nn.Linear(2, 64),    # Input → 64 hidden (Tanh)
        nn.Tanh(),
        nn.Linear(64, 2)     # 64 → 2 output classes
    )
```

**Architecture:** A single-hidden-layer network with 64 neurons and Tanh activation.

**Properties:**
- **Parameters:** $(2 \times 64 + 64) + (64 \times 2 + 2) = 192 + 130 = 322$ total parameters.
- **Robustness:** Tanh activation is bounded in $[-1, 1]$, naturally constraining output magnitudes and limiting the effect of adversarial noise on predictions. This produces lower output variance under perturbation.
- **Latency:** Slightly higher due to Tanh computation (vs. ReLU), though both are very fast.

#### 5.2.3 Design Rationale

The fragile model uses **ReLU** (unbounded, sharp) while the robust model uses **Tanh** (bounded, smooth). This ensures:

- Under low noise: both models perform similarly, but the fast model has slightly lower latency.
- Under high noise: the robust model's bounded activation naturally dampens the effect of perturbations on output variance, yielding higher reliability.

This asymmetry is the core of the model-selection problem.

---

### 5.3 Environment & Degradation

#### 5.3.1 Degradation Interface

**File:** `src/environment/degradation_interface.py`

```python
class DegradationProcess(abc.ABC):
    @abc.abstractmethod
    def get_noise(self, step): pass

    @abc.abstractmethod
    def get_latency_load(self, step): pass
```

All degradation processes implement this abstract base class, enabling polymorphic environment simulation.

#### 5.3.2 Structured Degradation (Default)

**File:** `src/environment/degradation.py`

**Class:** `EnvironmentDegradation`

This implements a **deterministic, multi-mode degradation pattern** composed of three superimposed effects:

##### Phase-Based Noise

$$
\text{base\_noise}(t) = \begin{cases}
0.01 & t < 150 \quad \text{(healthy phase)} \\
0.15 & 150 \le t < 300 \quad \text{(degraded phase)} \\
0.03 & t \ge 300 \quad \text{(recovery phase)}
\end{cases}
$$

##### Bursty Failures

Periodic bursts of high noise ($0.3$) occur every `burst_period` steps (default 100), lasting `burst_length` steps (default 10):

$$
\text{burst}(t) = \begin{cases}
0.3 & \text{if } t \bmod \text{burst\_period} < \text{burst\_length} \\
0.0 & \text{otherwise}
\end{cases}
$$

##### Gradual Drift

After `drift_start` (default step 200), noise increases linearly:

$$
\text{drift}(t) = \begin{cases}
0 & t < \text{drift\_start} \\
(t - \text{drift\_start}) \times \text{drift\_rate} & t \ge \text{drift\_start}
\end{cases}
$$

where `drift_rate` defaults to $0.0005$.

##### Adversarial Oscillation

After `osc_start` (default step 350), a sinusoidal oscillation is added:

$$
\text{osc}(t) = \begin{cases}
0 & t < \text{osc\_start} \\
A \cdot \sin\!\left(\frac{2\pi(t - \text{osc\_start})}{P}\right) & t \ge \text{osc\_start}
\end{cases}
$$

where $A = 0.12$ (amplitude) and $P = 20$ (period).

##### Combined Noise

$$
\text{noise}(t) = \text{clip}\!\left(\text{base\_noise}(t) + \text{drift}(t) + \text{osc}(t),\; 0,\; 1\right)
$$

Bursty noise overrides all other patterns when active.

##### Latency Model

Latency follows a similar structure:
- Base: 1000 computation units (healthy).
- Degraded: 10,000 units (during bursts, degraded phase 150–300, or oscillation peaks).

#### 5.3.3 Random Degradation (Unseen/Validation)

**File:** `src/environment/random_degradation.py`

**Class:** `RandomDegradation`

This generates **unpredictable noise and latency patterns** not used during controller tuning, for generalization testing.

##### Noise Model

A random walk with jumps and resets:

$$
n_{t+1} = \text{clip}(n_t + \Delta,\; 0.01,\; 0.25)
$$

where $\Delta \sim \text{Uniform}(-0.05, 0.05)$, with a 10% chance of an additional large jump $\Delta_{\text{jump}} \sim \text{Uniform}(-0.15, 0.15)$ and a 2% chance of reset to base (0.01).

##### Latency Model

A random walk with spikes:

$$
l_{t+1} = \text{clip}(l_t + s,\; 1000,\; 12000)
$$

where $s \sim \text{Uniform}(-500, 500)$, with a 5% chance of a spike $s_{\text{spike}} \sim \text{Uniform}(2000, 8000)$ and a 1% chance of reset to base (1000).

---

### 5.4 Fault Injection

**File:** `src/environment/injector.py`

**Class:** `FaultInjector`

The fault injector applies two types of faults probabilistically at each step:

1. **Latency Injection** (probability `latency_spike_prob`, default 0.15):
   - Simulates CPU-bound latency by performing heavy computation proportional to `env.get_latency_load(step)`.
   - This is a realistic simulation — actual wall-clock time increases.

2. **Noise Injection** (probability `noise_prob`, default 0.1):
   - Adds Gaussian noise: $x' = x + 0.2 \cdot \mathcal{N}(0, I)$.
   - Simulates data corruption or sensor noise.

**Parameters:**
- `latency_prob` (float): Probability of injecting latency per step.
- `noise_prob` (float): Probability of injecting noise per step.

---

### 5.5 Metrics

#### 5.5.1 Reliability Metric

**File:** `src/metrics/reliability.py`

**Class:** `ReliabilityMetric`

**Method:** `compute(model, x, noise_std, trials=10)`

Reliability quantifies how **consistently** a model produces the same output under input perturbation.

**Algorithm:**
1. Generate `trials` (default 10) copies of input $x$, each corrupted with Gaussian noise: $x_i = x + \epsilon_i$, where $\epsilon_i \sim \mathcal{N}(0, \sigma^2 I)$.
2. Run each through the model: $\hat{y}_i = f(x_i)$.
3. Compute the variance across trials: $V = \text{Var}(\{\hat{y}_i\}_{i=1}^{T})$.
4. Compute reliability as an exponential decay of variance:

$$
R = \exp(-\lambda \cdot V)
$$

where $\lambda = 50.0$ is a scaling constant.

5. Clip to $[0, 1]$.

**Interpretation:**
- $R \approx 1.0$: model produces consistent outputs despite noise (high reliability).
- $R \approx 0.0$: model outputs vary wildly with noise (low reliability).
- The exponential form ensures smooth, continuous degradation in the reliability signal.

**Why Variance?** Variance directly measures prediction instability. A model that is robust to noise will produce similar outputs regardless of perturbation, yielding low variance and high reliability.

#### 5.5.2 Latency Metric

**File:** `src/metrics/latency.py`

**Class:** `LatencyMetric`

**Method:** `measure(model, x)`

Measures wall-clock inference time using `time.perf_counter()`:

$$
L = t_{\text{end}} - t_{\text{start}}
$$

The measurement includes only forward-pass time (with `torch.no_grad()` for efficiency).

---

### 5.6 Smoothing

**File:** `src/metrics/smoothing.py`

**Class:** `ExponentialSmoother`

Implements **Exponential Weighted Moving Average (EWMA)**:

$$
S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}
$$

where:
- $S_t$ is the smoothed value at time $t$.
- $X_t$ is the raw observation at time $t$.
- $\alpha \in (0, 1)$ is the smoothing factor.
- $S_0 = X_0$ (initialization with first observation).

**Properties:**
- Higher $\alpha$ → more responsive to recent changes, less smoothing.
- Lower $\alpha$ → smoother signal, more resistance to transient noise.
- Default: $\alpha = 0.2$ for reliability, $\alpha = 0.2$ for latency (configurable).

**Effective window:** The EWMA approximates a simple moving average with window size $\frac{2}{\alpha} - 1$. For $\alpha = 0.2$, this is $\approx 9$ steps.

**Why EWMA?** Unlike a simple moving average, EWMA requires only $O(1)$ memory and computation per step, and gives exponentially decaying weight to older observations — ideal for real-time control loops.

---

### 5.7 Controllers

#### 5.7.1 DualSignalController (Main Controller)

**File:** `src/controller/dual_controller.py`

This is the **primary control algorithm** of the system.

**States:** Defined by the `StabilityState` enum:

| State | Value | Meaning |
|-------|-------|---------|
| `STABLE` | 0 | System operating normally with fast model |
| `DEGRADED` | 1 | Degradation detected, using robust model |
| `RECOVERING` | 2 | Transitioning back from robust to fast |
| `PREEMPTIVE_DEGRADED` | 3 | Predictive: degradation anticipated before crossing threshold |

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1.0 | Weight for reliability cost |
| `beta` | float | 1.0 | Weight for latency cost |
| `gamma` | float | 0.1 | Switch penalty coefficient |
| `horizon` | int | 1 | Prediction horizon (for future cost estimation) |
| `min_dwell_steps` | int | 0 | Minimum steps between switches (cooldown) |
| `osc_window` | int | 10 | Window size for oscillation detection |
| `osc_threshold` | int | 3 | Number of switches in window to trigger oscillation |
| `dwell_increase` | int | 10 | Additional dwell time when oscillating |

**Decision Algorithm:**

1. **Oscillation Detection:** Count switches in last `osc_window` steps. If ≥ `osc_threshold`, enter oscillation mode and increase dwell time.
2. **Multi-Objective Cost Computation:** For each model $m \in \{\text{fast}, \text{robust}\}$:

$$
J_m = \alpha \cdot (1 - R_m) + \beta \cdot L_m
$$

3. **Switch Penalty:** If the controller is considering a switch, add a penalty:

$$
J_{\text{switch}} = J_{\text{alt}} + \gamma \cdot P
$$

where $P$ is doubled if recent oscillation is detected ($\geq 2$ switches in last 4 steps).

4. **Decision Rule:**
   - Switch from fast → robust if: $J_{\text{robust}} + \gamma \cdot P < J_{\text{fast}}$
   - Switch from robust → fast if: $J_{\text{fast}} + \gamma \cdot P < J_{\text{robust}}$
   - Otherwise: hold current model.

5. **State Transition:**
   - fast → robust: state becomes `DEGRADED`.
   - robust → fast: state becomes `RECOVERING` → immediately `STABLE`.

#### 5.7.2 Baseline Controllers

**File:** `src/controller/baseline_controllers.py`

##### AlwaysFastController
Always selects the fast model. No decision logic. Serves as a lower bound for reliability.

##### AlwaysRobustController
Always selects the robust model. Serves as an upper bound for reliability but with higher latency.

##### ThresholdOnlyController
Switches based on **raw** (unsmoothed) reliability and latency against fixed thresholds:

$$
\text{action} = \begin{cases}
\text{robust} & \text{if } R < R_{\text{thresh}} \text{ or } L > L_{\text{thresh}} \\
\text{fast} & \text{otherwise}
\end{cases}
$$

No state machine, no hysteresis, no dwell time. Susceptible to oscillation.

##### SmoothingOnlyController
Same logic as ThresholdOnly, but operates on **smoothed** signals. Demonstrates the value of smoothing alone (without cost-based decision-making).

#### 5.7.3 LearningController (Q-Learning)

**File:** `src/controller/learning_controller.py`

An **online, tabular Q-learning controller** that learns a model-selection policy from experience.

**State Space:** A 6-tuple of discretized values:

$$
s = (\tilde{R},\; \tilde{L},\; \dot{R},\; \dot{L},\; O,\; C)
$$

where:
- $\tilde{R}$: smoothed reliability (rounded to 2 decimal places)
- $\tilde{L}$: smoothed latency (rounded to 2 decimal places)
- $\dot{R}$: reliability derivative
- $\dot{L}$: latency derivative
- $O$: oscillation score
- $C$: current controller state (enum value)

**Action Space:** $a \in \{\text{fast}, \text{robust}\}$

**Algorithm:** $\epsilon$-greedy Q-learning:

1. **Action Selection:**

$$
a_t = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s_t, a) & \text{otherwise}
\end{cases}
$$

2. **Q-Update (after observing reward $r_t$ and next state $s_{t+1}$):**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

**Reward Function:**

$$
r_t = -J_{a_t} - \gamma \cdot \mathbb{1}[\text{switch occurred}]
$$

where $J_{a_t}$ is the multi-objective cost of the chosen model and the switch penalty discourages oscillation.

**Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 0.1 | Exploration rate |
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.99 | Discount factor |

---

### 5.8 Monitoring & Telemetry

**File:** `src/monitoring/telemetry.py`

**Class:** `TelemetryLogger`

Collects per-step records into a Pandas DataFrame with timestamp. Each record includes:

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Simulation step |
| `reliability` | float | Raw reliability score |
| `smoothed_reliability` | float | EWMA-smoothed reliability |
| `latency` | float | Raw latency (seconds) |
| `smoothed_latency` | float | EWMA-smoothed latency |
| `active_model` | str | Currently active model ('fast' or 'robust') |
| `controller_state` | str | Controller state (STABLE, DEGRADED, etc.) |
| `deriv` | float | First derivative of smoothed reliability |
| `fast_J` | float | Multi-objective cost for fast model |
| `robust_J` | float | Multi-objective cost for robust model |
| `timestamp` | float | Unix timestamp |

---

### 5.9 Visualization

**File:** `src/visualization/plots.py`

**Function:** `plot_all(df, outdir)`

Generates four publication-quality plots per run:

1. **Reliability Plot** (`reliability.png`):
   - Raw reliability (translucent) vs. smoothed reliability (bold).
   - Marks predicted degradation (orange dashed) and actual degradation (red dotted) events.

2. **Latency Plot** (`latency.png`):
   - Raw latency (translucent) vs. smoothed latency (bold).

3. **Active Model Plot** (`active_model.png`):
   - Binary timeline: 0 = fast, 1 = robust.

4. **Controller State Plot** (`controller_state.png`):
   - Enum state value over time.

**Comparative Plots** (generated in `main.py`):
- `reliability_comparison.png`: All controllers on one reliability plot.
- `latency_comparison.png`: All controllers on one latency plot.
- `stability_*_comparison.png`: Bar charts for each formal stability metric.
- Per-controller `*_active_model.png` and `*_controller_state.png`.

---

### 5.10 Stability Metrics

**File:** `src/metrics/stability.py`

Three formal stability metrics are computed for rigorous controller evaluation:

#### Stability Horizon

The **longest consecutive period** where smoothed reliability stays above the threshold $\theta$:

$$
H = \max_{i,j} (j - i) \quad \text{s.t.} \quad \tilde{R}_t > \theta \;\; \forall t \in [i, j]
$$

Interpretation: Higher is better. A controller with $H = 500$ (full run) never dropped below threshold.

#### Oscillation Bound

The **maximum number of model switches** in any sliding window of size $W$ (default 50):

$$
B = \max_t \sum_{k=t}^{t+W-1} \mathbb{1}[m_k \neq m_{k-1}]
$$

Interpretation: Lower is better. A value of 1 means the controller switched at most once in any 50-step window. High values indicate chattering.

#### Recovery Time Distribution

For each episode where reliability drops below $\theta$, the number of steps to recover:

$$
\tau_i = \min\{t > t_{\text{drop},i} : \tilde{R}_t > \theta\} - t_{\text{drop},i}
$$

Reported as: mean, std, median, min, max, count. Lower mean recovery time is better.

---

## 6. Mathematical Foundations

### 6.1 Reliability as Prediction Stability

The core insight is that **reliability ≠ accuracy**. A model can be accurate on average but unreliable if its predictions are unstable under small perturbations. Reliability is formalized as:

$$
R(f, x, \sigma) = \exp\left(-\lambda \cdot \text{Var}_{{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}}\left[f(x + \epsilon)\right]\right)
$$

This is related to **local Lipschitz continuity**: a model with small Lipschitz constant around $x$ will have low output variance under perturbation and thus high reliability.

### 6.2 Exponential Smoothing Theory

EWMA is a special case of a **first-order IIR (Infinite Impulse Response) filter**. Its transfer function in the $z$-domain is:

$$
H(z) = \frac{\alpha}{1 - (1-\alpha)z^{-1}}
$$

The frequency response shows that EWMA acts as a **low-pass filter**, attenuating high-frequency noise while preserving the underlying trend.

**Half-life:** The number of steps for the weight of a past observation to decay to half:

$$
t_{1/2} = \frac{-\ln 2}{\ln(1 - \alpha)}
$$

For $\alpha = 0.2$: $t_{1/2} \approx 3.1$ steps.

### 6.3 Multi-Objective Cost Function

The controller minimizes a weighted sum of objectives:

$$
J(m, t) = \alpha \cdot \underbrace{(1 - R_m(t))}_{\text{unreliability}} + \beta \cdot \underbrace{L_m(t)}_{\text{latency}} + \gamma \cdot \underbrace{P(t)}_{\text{switch penalty}}
$$

This is a **scalarization** of a multi-objective optimization problem. The Pareto front is explored by varying $(\alpha, \beta, \gamma)$.

### 6.4 Hysteresis and Dwell Time

To prevent chattering (rapid back-and-forth switching), two mechanisms are used:

1. **Hysteresis:** The switch penalty $\gamma \cdot P$ creates a dead zone around the decision boundary. A switch only occurs if the alternative model is sufficiently better to overcome the penalty.

2. **Dwell Time:** After a switch, the controller is locked for `min_dwell_steps`. During oscillation, this is increased by `dwell_increase` steps (adaptive dwell).

The combination ensures **Zeno-free behavior** (no infinite switches in finite time).

### 6.5 Q-Learning Convergence

The Q-learning update converges to the optimal $Q^*$ under:

1. All state-action pairs are visited infinitely often.
2. The learning rate $\alpha_t$ satisfies $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$.
3. The reward is bounded.

With $\epsilon$-greedy exploration ($\epsilon = 0.1$), all state-action pairs are eventually visited. The fixed learning rate ($\alpha = 0.1$) does not satisfy the theoretical conditions but works well empirically for non-stationary environments.

---

## 7. Control Theory & State Machine

### 7.1 State Machine Diagram

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
    ┌─────────┐  J_robust < J_fast  ┌──────────┴──┐
    │ STABLE  │─────────────────────▶│  DEGRADED   │
    │ (fast)  │                      │  (robust)   │
    └────┬────┘                      └──────┬──────┘
         ▲                                  │
         │        J_fast < J_robust         │
         │  ┌────────────┐                  │
         └──│ RECOVERING │◀─────────────────┘
            └────────────┘
                  │
                  │ (immediate transition)
                  ▼
            ┌─────────┐
            │ STABLE  │
            └─────────┘

    ┌─────────────────────────────────────────────┐
    │           PREEMPTIVE_DEGRADED               │
    │  (triggered by predictive derivative        │
    │   analysis, overrides STABLE → DEGRADED)    │
    └─────────────────────────────────────────────┘
```

### 7.2 Transition Conditions

| From | To | Condition |
|------|----|-----------|
| STABLE | DEGRADED | $J_{\text{robust}} + P < J_{\text{fast}}$ and dwell satisfied |
| DEGRADED | RECOVERING | $J_{\text{fast}} + P < J_{\text{robust}}$ and dwell satisfied |
| RECOVERING | STABLE | Immediate (same step as RECOVERING) |
| Any | PREEMPTIVE_DEGRADED | Persistent negative derivative detected |
| PREEMPTIVE_DEGRADED | DEGRADED | Reliability actually crosses threshold |

### 7.3 Oscillation Detection

A sliding window of size `osc_window` tracks recent switch/hold events. If the number of switches exceeds `osc_threshold`:

1. `oscillating` flag is set to `True`.
2. `min_dwell_steps` is increased by `dwell_increase`.
3. When oscillation ceases, `stabilization_time` is recorded.

This implements an **adaptive backoff** similar to congestion control in networking.

---

## 8. Multi-Objective Optimization

### 8.1 Formulation

At each step $t$, the system faces a bi-objective optimization:

$$
\min_{m \in \{F, R\}} \left( 1 - R_m(t),\; L_m(t) \right)
$$

This is scalarized into:

$$
J_m(t) = \alpha (1 - R_m(t)) + \beta L_m(t)
$$

### 8.2 Switch Cost as Regularization

Adding the switch penalty transforms the problem into:

$$
J_m^{\text{total}}(t) = J_m(t) + \gamma \cdot P(t) \cdot \mathbb{1}[m \neq m_{t-1}]
$$

This is analogous to **L1 regularization** on the switch frequency — encouraging sparse switching.

### 8.3 Parameter Sensitivity

| $\alpha$ | $\beta$ | $\gamma$ | Effect |
|----------|---------|----------|--------|
| High | Low | Low | Prioritize reliability, switch aggressively |
| Low | High | Low | Prioritize latency, prefer fast model |
| Any | Any | High | Strong switching penalty, more conservative |
| 1.0 | 1.0 | 0.1 | **Default**: balanced with mild switch penalty |

---

## 9. Reinforcement Learning Controller

### 9.1 MDP Formulation

| Element | Definition |
|---------|-----------|
| **State** | $s = (\tilde{R}, \tilde{L}, \dot{R}, \dot{L}, O, C)$ — smoothed signals, derivatives, oscillation score, controller state |
| **Action** | $a \in \{\text{fast}, \text{robust}\}$ |
| **Reward** | $r = -J_a - \gamma \cdot \mathbb{1}[\text{switch}]$ |
| **Transition** | Determined by environment (non-stationary) |

### 9.2 State Discretization

All continuous signals are rounded to 2 decimal places for tabular Q-learning. This creates a large but finite state space.

### 9.3 Exploration vs. Exploitation

With $\epsilon = 0.1$, the controller explores 10% of the time. This ensures it discovers beneficial switching policies but may cause higher oscillation during exploration (observed: oscillation bound of 40–43 vs. 3–6 for the main controller).

### 9.4 Comparison with Main Controller

The learning controller is **adaptive** — it can theoretically learn optimal policies for any environment. However, in the 500-step horizon:
- It shows **higher oscillation** due to exploration.
- It achieves **competitive reliability** in random environments.
- It requires **longer episodes** to converge fully.

---

## 10. Predictive Degradation Detection

### 10.1 Derivative-Based Early Warning

The system computes the first-order derivative of smoothed reliability:

$$
\dot{R}_t = \tilde{R}_t - \tilde{R}_{t-1}
$$

A rolling mean of the derivative over a configurable window (`predictive_deriv_window`, default 5) is monitored. If the mean derivative is persistently negative (below `predictive_neg_deriv_thresh`, default $-0.002$) for `predictive_neg_trend_steps` (default 3) consecutive windows, the system enters `PREEMPTIVE_DEGRADED` state.

### 10.2 Volatility-Based Confirmation

Rolling volatility (standard deviation) of smoothed reliability over a window (`predictive_vol_window`, default 10) is also checked. High volatility ($> 0.01$) combined with negative trend confirms impending degradation.

### 10.3 Lead Time

The system records:
- `predicted_degradation_step`: when preemptive detection triggers.
- `actual_degradation_step`: when reliability actually crosses the threshold.
- `lead_time = actual - predicted`: how far in advance the system detected degradation.

---

## 11. Configuration Reference

**File:** `self_stabilizing_inference/config/config.yaml`

```yaml
# ============================================
# COMPLETE CONFIGURATION REFERENCE
# ============================================

random_seed: 42                    # Global random seed for reproducibility

inference:
  batch_size: 32                   # Inference batch size (unused in current loop)

reliability:
  window_size: 50                  # Window for reliability computation (unused, trials used instead)

degradation:
  ewma_alpha: 0.2                  # Smoothing factor for reliability EWMA
  threshold: 0.65                  # Reliability threshold for degradation detection
  latency_ewma_alpha: 0.2         # Smoothing factor for latency EWMA (default)
  latency_threshold: 0.1          # Latency threshold for switching (default)
  # Structured degradation parameters:
  burst_period: 100                # Steps between burst events
  burst_length: 10                 # Duration of each burst
  burst_noise: 0.3                 # Noise level during bursts
  drift_start: 200                 # Step when gradual drift begins
  drift_rate: 0.0005               # Rate of linear noise drift
  osc_start: 350                   # Step when adversarial oscillation begins
  osc_period: 20                   # Period of sinusoidal oscillation
  osc_amplitude: 0.12              # Amplitude of oscillation
  base_noise: 0.01                 # Noise in healthy phase
  degraded_noise: 0.15             # Noise in degraded phase
  recovered_noise: 0.03            # Noise in recovery phase
  base_latency: 1000               # Computation units in healthy state
  degraded_latency: 10000          # Computation units in degraded state
  # Predictive detection parameters:
  predictive_deriv_window: 5       # Window for mean derivative
  predictive_vol_window: 10        # Window for rolling volatility
  predictive_neg_deriv_thresh: -0.002  # Threshold for negative mean derivative
  predictive_vol_thresh: 0.01      # Threshold for volatility
  predictive_neg_trend_steps: 3    # Consecutive steps of negative trend

control:
  alpha: 1.0                       # Reliability cost weight
  beta: 1.0                        # Latency cost weight
  gamma: 0.1                       # Switch penalty coefficient
  horizon: 1                       # Prediction horizon
  cooldown_steps: 30               # Minimum dwell time between switches

faults:
  latency_spike_prob: 0.15         # Probability of latency fault per step
  noise_prob: 0.1                  # Probability of noise fault per step

learning_controller:
  epsilon: 0.1                     # Exploration rate for Q-learning
  alpha: 0.1                       # Learning rate for Q-learning
  gamma: 0.99                      # Discount factor for Q-learning
```

---

## 12. Data Flow — End-to-End Pipeline

### Step-by-Step Execution (per simulation step)

```
Step t:
│
├─ 1. Generate random input: x ~ Uniform(-1, 1)²
│
├─ 2. Inject faults: x' = FaultInjector.inject(x, step, env)
│      ├─ 15% chance: simulate CPU latency (heavy computation)
│      └─ 10% chance: add Gaussian noise (σ=0.2)
│
├─ 3. Get environment noise: σ_env = env.get_noise(step)
│      (Phase-based + burst + drift + oscillation OR random walk)
│
├─ 4. Compute metrics for BOTH models:
│      ├─ fast_reliability  = ReliabilityMetric.compute(fast_model,  x', σ_env)
│      ├─ fast_latency      = LatencyMetric.measure(fast_model,  x')
│      ├─ robust_reliability = ReliabilityMetric.compute(robust_model, x', σ_env)
│      └─ robust_latency     = LatencyMetric.measure(robust_model, x')
│
├─ 5. Select active model's metrics:
│      reliability = fast_reliability if active=='fast' else robust_reliability
│      latency     = fast_latency     if active=='fast' else robust_latency
│
├─ 6. Smooth signals:
│      smoothed_reliability = EWMA(reliability)
│      smoothed_latency     = EWMA(latency)
│
├─ 7. Compute derivative:
│      deriv = smoothed_reliability[t] - smoothed_reliability[t-1]
│
├─ 8. Controller decision:
│      action, new_state, oscillating, stab_time = controller.decide(
│          smoothed_reliability, smoothed_latency, state, step, ...)
│
├─ 9. Apply action:
│      if action: active_model = action
│      stability_state = new_state
│
└─ 10. Log telemetry:
       logger.log({step, reliability, smoothed_reliability, latency,
                   smoothed_latency, active_model, controller_state,
                   deriv, fast_J, robust_J})
```

### Outer Loop Structure

```
For each environment ∈ {default, random}:
    For each controller ∈ {always_fast, always_robust, threshold_only,
                           smoothing_only, main, learning}:
        Reset seeds (identical conditions)
        Reset environment, metrics, smoother
        Run 500 steps
        Store results
Save CSVs
Generate plots
Compute stability metrics
Save stability summary
```

---

## 13. Iterative Development History

The system was developed across 8 iterations, each adding new capabilities:

### Iteration 1: Basic Dual-Signal Control
- Implemented basic fast/robust model switching.
- Added reliability and latency measurement.
- Simple threshold-based control with EWMA smoothing.
- **Output:** Basic reliability/latency/model/state plots.

### Iteration 2: Adaptive Smoothing
- Made EWMA $\alpha$ configurable.
- Experimented with different smoothing levels.
- Observed trade-off between responsiveness and stability.

### Iteration 3: Predictive Degradation Detection
- Added first-order derivative analysis of smoothed reliability.
- Implemented rolling volatility monitoring.
- Added `PREEMPTIVE_DEGRADED` state.
- Logged predicted vs. actual degradation steps and lead time.

### Iteration 4: Multi-Objective Controller
- Replaced simple threshold logic with multi-objective cost function.
- Introduced weighted sum $J = \alpha(1-R) + \beta L + \gamma P$.
- Enabled simultaneous optimization of reliability and latency.
- **Output:** Plots showing cost-driven switching behavior.

### Iteration 5: Oscillation Detection & Adaptive Dwell
- Added sliding-window oscillation detection.
- Implemented adaptive dwell time increase during oscillation.
- Recorded stabilization time after oscillation episodes.

### Iteration 6: Baseline Controllers & Comparative Evaluation
- Implemented four baseline controllers (Always Fast, Always Robust, Threshold Only, Smoothing Only).
- Enabled side-by-side comparison under identical conditions (seed-controlled).
- **Output:** Comparative reliability/latency plots.

### Iteration 7: Advanced Environment & Robustness Metrics
- Added structured degradation with bursty + drift + oscillation patterns.
- Implemented formal stability metrics (horizon, oscillation bound, recovery time).
- First formal controller comparison summary.
- **Output:** Controller comparison summary CSV and comparative plots.

### Iteration 8: Learning Controller, Random Environment, Full Evaluation
- Implemented Q-learning `LearningController`.
- Added `RandomDegradation` environment for generalization testing.
- Full cross-product evaluation: 6 controllers × 2 environments.
- Complete stability metric comparison with bar charts.
- **Output:** 12 metric CSVs, 29+ plots, stability summary CSV.

---

## 14. Experimental Results and Analysis

### 14.1 Controller Comparison — Default Environment (Iteration 7)

| Controller | Avg Reliability | Avg Latency | Oscillation Count | Recovery Time | Max Stability Duration |
|------------|:-:|:-:|:-:|:-:|:-:|
| always_fast | 0.9283 | 4.22e-05 | 1 | 1.0 | 499 |
| always_robust | 0.9589 | 2.22e-05 | 1 | — | 500 |
| threshold_only | 0.9290 | 2.40e-05 | 16 | 1.0 | 499 |
| smoothing_only | 0.9280 | 2.41e-05 | 2 | 1.0 | 499 |
| main (dual-signal) | 0.9567 | 3.05e-05 | 5 | 1.0 | 499 |

**Key Observations:**
- The **main dual-signal controller** achieves reliability (0.9567) close to always-robust (0.9589) while using the fast model when possible.
- The **threshold-only controller** oscillates 16 times — demonstrating why raw thresholds without smoothing or cost-based decisions are inadequate.
- **Smoothing-only** reduces oscillation to 2 but has lower reliability than the main controller.
- The main controller achieves a **3.2× reduction** in oscillation vs. threshold-only.

### 14.2 Full Cross-Environment Comparison (Iteration 8)

#### Default Environment

| Controller | Avg Reliability | Avg Latency | Stability Horizon | Oscillation Bound | Recovery Time Mean |
|------------|:-:|:-:|:-:|:-:|:-:|
| always_fast | 0.9283 | 4.31e-05 | 499 | 1 | 1.0 |
| always_robust | 0.9589 | 2.19e-05 | 500 | 1 | — |
| threshold_only | 0.9290 | 2.35e-05 | 499 | 6 | 1.0 |
| smoothing_only | 0.9280 | 2.37e-05 | 499 | 2 | 1.0 |
| main | 0.9567 | 2.27e-05 | 499 | 3 | 1.0 |
| learning | 0.9454 | 2.33e-05 | 499 | 43 | 1.0 |

#### Random (Unseen) Environment

| Controller | Avg Reliability | Avg Latency | Stability Horizon | Oscillation Bound | Recovery Time Mean |
|------------|:-:|:-:|:-:|:-:|:-:|
| always_fast | 0.9565 | 2.37e-05 | 500 | 1 | — |
| always_robust | 0.9745 | 2.19e-05 | 500 | 1 | — |
| threshold_only | 0.9388 | 2.35e-05 | 500 | 4 | — |
| smoothing_only | 0.9675 | 2.34e-05 | 500 | 1 | — |
| main | **0.9763** | 2.22e-05 | 500 | 4 | — |
| learning | 0.9661 | 2.28e-05 | 500 | 40 | — |

### 14.3 Analysis and Inferences

#### Inference 1: Main Controller Achieves Near-Optimal Reliability
The main dual-signal controller (0.9567 default, **0.9763 random**) consistently matches or exceeds the always-robust controller in the random environment, demonstrating that the multi-objective cost function effectively identifies when switching is beneficial.

#### Inference 2: Smoothing Dramatically Reduces Oscillation
Comparing threshold-only (oscillation bound 6) vs. smoothing-only (bound 2) shows that EWMA smoothing alone cuts oscillation by 3×. The main controller (bound 3) adds cost-based decision-making on top.

#### Inference 3: Learning Controller Has High Oscillation
The learning controller's oscillation bound (40–43) is 10× higher than the main controller. This is due to $\epsilon = 0.1$ exploration — 10% of actions are random, causing frequent unnecessary switches. This is a known trade-off of online RL: exploration harms short-term performance.

#### Inference 4: Generalization to Unseen Environments
In the random environment, the main controller achieves **higher reliability than always-robust** (0.9763 vs. 0.9745). This suggests the controller's switching logic provides an additional benefit: by selecting the best model for each moment, it can outperform any single model across diverse conditions.

#### Inference 5: Stability Horizon is Universally High
All controllers achieve stability horizons of 499–500, indicating that the reliability threshold (0.65) is rarely breached. This validates the threshold setting — it's low enough to be achievable but high enough to be meaningful.

#### Inference 6: Recovery is Immediate
Recovery time mean is uniformly 1.0 where applicable, indicating that the system recovers within one step of switching to the robust model. This confirms that the robust model is genuinely robust to the degradation levels in both environments.

### 14.4 Sample Per-Step Data (Main Controller, Default Env)

| Step | Reliability | Smoothed R | Latency | Smoothed L | Active Model | State | Deriv | Fast J | Robust J |
|---:|:--:|:--:|:--:|:--:|:---|:---|:--:|:--:|:--:|
| 0 | 0.646 | 0.646 | 6.94e-5 | 6.94e-5 | robust | DEGRADED | 0.000 | 0.354 | 0.204 |
| 1 | 0.770 | 0.671 | 2.46e-5 | 6.05e-5 | fast | STABLE | 0.025 | 0.083 | 0.230 |
| 2 | 0.832 | 0.703 | 2.51e-5 | 5.34e-5 | fast | STABLE | 0.032 | 0.168 | 0.194 |
| 3 | 0.813 | 0.725 | 2.88e-5 | 4.85e-5 | robust | DEGRADED | 0.022 | 0.187 | 0.085 |
| 4 | 0.909 | 0.762 | 2.23e-5 | 4.32e-5 | robust | DEGRADED | 0.037 | 0.106 | 0.091 |

This shows the controller initially switching between models as reliability builds up, then stabilizing on the robust model during early uncertainty.

---

## 15. Visualizations and Plot Inventory

### 15.1 Iteration 1 Plots

| Plot | Path | Description |
|------|------|-------------|
| Reliability | `src/plots/iteration_1/20260203_211648/reliability.png` | Raw vs. smoothed reliability |
| Latency | `src/plots/iteration_1/20260203_211648/latency.png` | Raw vs. smoothed latency |
| Active Model | `src/plots/iteration_1/20260203_211648/active_model.png` | Binary model selection timeline |
| Controller State | `src/plots/iteration_1/20260203_211648/controller_state.png` | State transitions over time |

**Interpretation:** These show the basic system behavior — the controller detects the degraded phase (steps 150–300), switches to robust, and switches back during recovery.

### 15.2 Iteration 4 Plots

Located in `src/plots/iteration_4/` with three runs. Same plot types as Iteration 1, but with the multi-objective cost function driving decisions.

**Interpretation:** Model switching is now driven by cost comparison rather than simple thresholds, resulting in fewer unnecessary switches.

### 15.3 Iteration 7 Plots

| Plot | Path |
|------|------|
| Reliability Comparison | `src/plots/iteration_7/20260203_213048/reliability_comparison.png` |
| Latency Comparison | `src/plots/iteration_7/20260203_213048/latency_comparison.png` |

**Interpretation:** First side-by-side comparison of all controllers. Shows that the main controller's smoothed reliability closely tracks the always-robust baseline, while threshold-only shows visible oscillations.

### 15.4 Iteration 8 Plots (Final, Most Comprehensive)

Located in `src/plots/iteration_8/20260203_220313/`:

#### Comparative Plots
| Plot | Description |
|------|-------------|
| `reliability_comparison.png` | All 12 controller-environment combinations' smoothed reliability |
| `latency_comparison.png` | All 12 combinations' smoothed latency |
| `stability_stability_horizon_comparison.png` | Bar chart: stability horizon by controller and environment |
| `stability_oscillation_bound_comparison.png` | Bar chart: oscillation bound by controller and environment |
| `stability_recovery_time_mean_comparison.png` | Bar chart: mean recovery time by controller and environment |

#### Per-Controller Plots (for each of 12 combinations)
| Pattern | Description |
|---------|-------------|
| `{ctrl}_{env}_active_model.png` | Model selection timeline |
| `{ctrl}_{env}_controller_state.png` | State transitions |

**Total:** 29 plots in iteration 8 alone.

**Key Visual Observations:**
- **Oscillation Bound Chart:** The learning controller bar is dramatically taller (40–43) than all others (1–6), visually highlighting the exploration cost.
- **Reliability Comparison:** All controllers except threshold-only maintain smooth reliability curves. Threshold-only shows jagged oscillations.
- **Active Model Plots:** The main controller shows sparse, decisive switches. The learning controller shows frequent alternation.

---

## 16. Conclusions & Key Inferences

### 16.1 Primary Conclusions

1. **Dual-signal control is effective.** The multi-objective cost function, combined with EWMA smoothing and oscillation-aware switching, achieves near-optimal reliability with minimal oscillation.

2. **Smoothing is essential.** Without EWMA smoothing, threshold-based controllers oscillate 3–10× more. Smoothing provides temporal stability to the decision-making process.

3. **The switch penalty prevents chattering.** The $\gamma \cdot P$ term, combined with adaptive dwell time, ensures the controller does not react to every fluctuation.

4. **The system generalizes to unseen environments.** The main controller achieves its **best performance** in the random (unseen) environment, demonstrating that the control logic is not overfit to the structured degradation pattern.

5. **Reinforcement learning is promising but requires tuning.** The Q-learning controller achieves competitive reliability but suffers from high oscillation due to exploration. Reducing $\epsilon$ or using decay would improve this.

### 16.2 Design Recommendations

- **Use $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.1$** as defaults for balanced performance.
- **Set `cooldown_steps = 30`** to prevent rapid switching.
- **Use EWMA $\alpha = 0.2$** for a good balance of responsiveness and smoothness.
- **The reliability threshold of 0.65** is appropriate for this model pair and noise range.
- **For production:** consider decaying $\epsilon$ in the learning controller and running longer episodes for convergence.

### 16.3 Theoretical Contributions

- Formalization of reliability as prediction stability (variance-based).
- Multi-objective cost formulation for model selection.
- Formal stability metrics (horizon, oscillation bound, recovery time).
- Demonstration that control-theoretic techniques (hysteresis, dwell time, adaptive backoff) are effective for ML inference stabilization.

---

## 17. Dual Codebase Explanation

This project contains **two codebases**:

### `self_stabilizing_inference/` — Legacy Prototype

- Uses **Keras/TensorFlow** for models (Dense layers).
- Implements a **simple control loop** with hardcoded thresholds (`LOW_THRESHOLD = 0.40`, `HIGH_THRESHOLD = 0.70`).
- Single controller (cooldown-based).
- Basic EWMA smoothing (manual computation).
- Reliability uses `model.predict()` (Keras API).
- Fault injection uses `time.sleep()` (real delay).
- Contains `InferenceEngine` with entropy/confidence metrics.
- Serves as the **proof-of-concept** that established the core idea.

### `src/` — Primary Advanced System

- Uses **PyTorch** for models.
- Implements **6 controllers** with formal state machines.
- Multi-objective cost function with switch penalty.
- Oscillation detection and adaptive dwell time.
- Q-learning RL controller.
- Two environment modes (structured + random).
- Formal stability metrics.
- Comprehensive logging, CSV export, and visualization.
- **This is the authoritative, production-ready codebase.**

---

## 18. How to Reproduce

### 18.1 Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PyYAML

### 18.2 Installation

```bash
git clone <repository-url>
cd Self-Stabilizing-ML-Inference-System
pip install torch numpy pandas matplotlib pyyaml
```

### 18.3 Running the Experiment

```bash
cd src
python main.py
```

This will:
1. Load configuration from `self_stabilizing_inference/config/config.yaml`.
2. Run all 6 controllers under both default and random environments (500 steps each).
3. Save per-step metrics to `src/results/metrics/iteration_8/{timestamp}/`.
4. Save comparative plots to `src/plots/iteration_8/{timestamp}/`.
5. Print the stability metrics summary to stdout.

### 18.4 Running the Legacy Prototype

```bash
cd self_stabilizing_inference
python main.py
```

This runs the simpler Keras/TF version with hardcoded thresholds and a single controller.

### 18.5 Adjusting Configuration

Edit `self_stabilizing_inference/config/config.yaml` to modify:
- Smoothing parameters (`ewma_alpha`).
- Degradation patterns (bursts, drift, oscillation).
- Controller cost weights ($\alpha$, $\beta$, $\gamma$).
- Fault injection probabilities.
- Learning controller hyperparameters.

---

## 19. Extending the System

### 19.1 Adding a New Controller

1. Create a new file in `src/controller/`.
2. Implement a class with a `decide(...)` method returning `(action, new_state, oscillating, stabilization_time)`.
3. Add an instance to the `baseline_controllers` dict in `src/main.py`.
4. The controller will automatically be included in all comparative evaluations.

### 19.2 Adding a New Degradation Pattern

1. Create a new class in `src/environment/` that extends `DegradationProcess`.
2. Implement `get_noise(step)` and `get_latency_load(step)`.
3. Add a new `env_mode` case in the outer loop of `src/main.py`.

### 19.3 Adding a New Model

1. Create a new PyTorch `nn.Module` in `src/models/`.
2. Instantiate it in `src/main.py`.
3. Expand the action space of controllers if needed (currently binary: fast/robust).

### 19.4 Adding New Metrics

1. Add a new metric class in `src/metrics/`.
2. Compute it in the step loop and include it in the telemetry log record.
3. Add visualization in `src/visualization/plots.py`.

---

## 20. Known Limitations & Future Work

### 20.1 Current Limitations

1. **Models are untrained.** Both models use random weights — reliability differences come from architecture, not learned features. Adding training would make results more realistic.
2. **Fixed 500-step horizon.** The learning controller may not converge in 500 steps. Longer episodes or pre-training would help.
3. **No real data.** The XOR-like synthetic data does not represent real-world distributions.
4. **Single-sample inference.** Each step processes one sample, not a batch. Batch inference would better simulate production.
5. **No model confidence calibration.** Reliability is based on output variance, not calibrated probabilities.

### 20.2 Future Directions

1. **Train models on real datasets** (CIFAR-10, medical imaging, NLP tasks).
2. **Deep RL controller** (DQN, PPO) instead of tabular Q-learning.
3. **Multi-model pool** (>2 models) with graduated capability levels.
4. **Online model retraining** when degradation is detected.
5. **Distributed deployment** with per-node controllers.
6. **Formal verification** of stability guarantees using Lyapunov analysis.
7. **Integration with MLOps platforms** (Kubernetes, KServe).

---

## 21. Glossary

| Term | Definition |
|------|-----------|
| **EWMA** | Exponential Weighted Moving Average — a low-pass filter for time-series smoothing |
| **Reliability** | $\exp(-\lambda \cdot \text{Var}(\text{predictions}))$ — measure of prediction consistency under noise |
| **Latency** | Wall-clock inference time (seconds) |
| **Dwell Time** | Minimum number of steps between model switches (cooldown) |
| **Hysteresis** | Dead zone created by switch penalty to prevent oscillation |
| **Oscillation Bound** | Maximum model switches in any sliding window |
| **Stability Horizon** | Longest consecutive period above reliability threshold |
| **Recovery Time** | Steps required to return above threshold after a drop |
| **Multi-Objective Cost** | $J = \alpha(1-R) + \beta L + \gamma P$ — weighted sum of reliability loss, latency, and switch penalty |
| **Q-Learning** | Tabular reinforcement learning algorithm for learning action-value functions |
| **$\epsilon$-Greedy** | Exploration strategy: random action with probability $\epsilon$, best known action otherwise |
| **Chattering** | Rapid, oscillatory switching between models |
| **Zeno Behavior** | Infinite switches in finite time (prevented by dwell time) |
| **Scalarization** | Converting a multi-objective problem to single-objective via weighted sum |
| **IIR Filter** | Infinite Impulse Response — EWMA is a first-order IIR filter |
| **Pareto Front** | Set of non-dominated solutions in multi-objective optimization |
| **Lipschitz Continuity** | Bounded rate of change — relevant to reliability analysis |

---

## 22. Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 0.1 | Initial prototype (`self_stabilizing_inference/`) |
| 2026-02-03 | 0.2 | Iteration 1: Basic dual-signal control (PyTorch) |
| 2026-02-03 | 0.3 | Iteration 2: Adaptive smoothing |
| 2026-02-03 | 0.4 | Iteration 3: Predictive degradation detection |
| 2026-02-03 | 0.5 | Iteration 4: Multi-objective controller |
| 2026-02-03 | 0.6 | Iteration 5: Oscillation detection & adaptive dwell |
| 2026-02-03 | 0.7 | Iteration 6: Baseline controllers & comparative evaluation |
| 2026-02-03 | 0.8 | Iteration 7: Advanced environment & formal stability metrics |
| 2026-02-03 | 0.9 | Iteration 8: Learning controller, random environment, full evaluation |
| 2026-02-13 | 1.0 | Comprehensive project manual created |

---

> **Remember:** This document is a **living document**. Update it with every change to the codebase. The next contributor should be able to understand the entire system by reading only this file.

---

*End of Project Manual*
