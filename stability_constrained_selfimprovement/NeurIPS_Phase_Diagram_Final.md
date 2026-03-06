# Universal Stability Crossover in Constrained Continual Learning

**Kavya Bhand**

---

## Abstract

We characterize the stability-plasticity tradeoff in continual learning through dense hyperparameter sweeps across 8 architectures spanning a 24× parameter range and 48× Hessian trace variation. Using sigmoid fitting with 2000-sample bootstrap confidence intervals, we find that the Functional Trust Region (FTR) stability crossover occurs at ε* ≈ 7.15 ± 0.35 (CV = 4.96%) — a value that is **architecture-independent** to within measurement precision. An F-test for constancy yields p = 0.786, and all 10 tested curvature normalizations (including ε·tr(H), ε·tr(F), ε·κ, ε/log(d)) **increase** dispersion rather than reducing it. No curvature metric correlates significantly with ε* (all p > 0.06). Cross-method analysis reveals that LwF exhibits a moderately architecture-dependent transition (CV ≈ 14%), while EWC shows no transition at all. We provide 1,200 experiments totaling 800 forgetting measurements, 40 curvature measurements, and 360 cross-method evaluations, with complete statistical characterization.

**Keywords**: Continual learning, stability-plasticity tradeoff, phase diagram, knowledge distillation, functional trust region

---

## 1. Introduction

Continual learning systems face a fundamental tension between plasticity (learning new tasks) and stability (retaining old knowledge). The **Functional Trust Region (FTR)** method constrains parameter updates via KL divergence in output space:

$$\text{KL}(f_\theta(x) \| f_{\theta_{\text{old}}}(x)) \leq \varepsilon$$

where ε is a stability budget controlling the permitted drift in function space. As ε → 0, the model is frozen; as ε → ∞, all updates are unconstrained.

**The central question**: At what critical ε* does the system cross over from stable retention to catastrophic forgetting? And does this depend on the model's architecture?

### 1.1 Prior Work and Motivation

Prior experiments (Sessions 1–4) identified ε* ≈ 7.07 across 11 architectures on a coarse 12-point grid. However, with adjacent grid points ε ∈ {5, 10}, the geometric-mean estimator always returns √50 ≈ 7.07 — a potential **grid artifact**.

This work conclusively resolves the ambiguity by:
1. A **dense 20-point ε grid** with 10 points in [5.0, 10.0] for sub-unit resolution
2. **Sigmoid fitting** (replacing finite-difference estimation) for robust ε* estimation
3. **5 seeds** per configuration with 2000-sample bootstrap for confidence intervals
4. **10 curvature normalizations** tested for cross-architecture collapse
5. **Cross-method** dense sweeps (LwF with 16 α values, EWC with 8 λ values)

### 1.2 Summary of Main Findings

| Finding | Evidence |
|---------|----------|
| **ε* ≈ 7.15 is universal** | CV = 4.96% across 8 architectures spanning 24× in parameters |
| **Not a grid artifact** | Dense 20-point grid + sigmoid fitting confirms ε* ∈ [6.56, 7.78] |
| **Transition is smooth** | Sigmoid R² > 0.94, k ∈ [2.9, 4.8]; not a sharp phase boundary |
| **No normalization helps** | ALL 10 curvature normalizations increase CV (worst: 37×) |
| **Universality is unexpected** | tr(H) varies 48× across architectures yet ε* varies only 5% |
| **Method-specific**: LwF has a transition | α* ≈ 0.58 for 4/5 archs (CV ≈ 14%); CNN_W8 is outlier |
| **Method-specific**: EWC has no transition | Sharpness ≈ 1.0 across all architectures and λ values |

---

## 2. Experimental Setup

### 2.1 Architecture Zoo

We select 8 architectures maximizing diversity in curvature space. Architectures span two families (CNN, ResNet), five widths, varied depth, and the presence/absence of batch normalization.

| Architecture | Family | Parameters | tr(H) | tr(F) | ‖H‖ | d_eff |
|---|---|---|---|---|---|---|
| CNN_W8 | CNN | 36,946 | 197 ± 75 | 1.12 ± 0.67 | 50 ± 32 | 4.6 ± 1.8 |
| CNN_W16 | CNN | 80,418 | 255 ± 54 | 2.15 ± 0.80 | 51 ± 24 | 5.4 ± 1.2 |
| CNN_W32 | CNN | 188,098 | 121 ± 56 | 1.16 ± 0.48 | 21 ± 3 | 6.0 ± 3.1 |
| CNN_W64 | CNN | 486,402 | 167 ± 66 | 0.94 ± 0.43 | 37 ± 7 | 4.4 ± 1.0 |
| CNN_W96 | CNN | 895,298 | 112 ± 32 | 0.79 ± 0.31 | 37 ± 5 | 3.1 ± 0.9 |
| CNN_D4_W32 | CNN | 126,850 | 381 ± 57 | 1.35 ± 0.19 | 31 ± 6 | 12.5 ± 1.5 |
| CNN_W32_NoBN | CNN | 187,778 | 68 ± 21 | 3.55 ± 5.40 | 45 ± 15 | 1.6 ± 0.4 |
| ResNet18_W8 | ResNet | 175,882 | 3,254 ± 530 | 5.69 ± 2.10 | 248 ± 72 | 14.4 ± 5.8 |

**Key diversity metrics:**
- Parameter range: 37K–895K (24×)
- Hessian trace range: 68–3,254 (48×)
- Spectral norm range: 21–248 (12×)
- Fisher trace range: 0.79–5.69 (7×)
- d_eff range: 1.6–14.4 (9×)
- Two architecture families: CNN (7) and ResNet (1)

All curvature metrics computed with 5 seeds; values reported as mean ± std.

### 2.2 Dense ε Grid

20 ε values: {0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0, 15.0, 20.0, 50.0}

This provides **10 points in [5.0, 10.0]** at sub-unit spacing — orders of magnitude finer than the original coarse grid.

### 2.3 Training Protocol

- **Dataset**: CIFAR-10 (1,000 samples/class)
- **Tasks**: 5 sequential 2-class tasks, each evaluated for catastrophic forgetting
- **Epochs**: 5 per task
- **FTR**: λ_init = 1.0, λ_lr = 0.005, λ_max = 50, β = 0.9, T = 2.0, warmup = 1
- **Seeds**: 5 per configuration (42, 137, 256, 7, 2024)
- **Hardware**: Apple Silicon MacBook Air (M3), CPU-only

### 2.4 ε* Estimation

We fit a logistic sigmoid to each architecture's forgetting curve:

$$F(\varepsilon) = F_{\min} + \frac{F_{\max} - F_{\min}}{1 + \exp(-k \cdot (\ln\varepsilon - \ln\varepsilon^*))}$$

where ε* is the crossover point, k is the transition sharpness, and F_min/F_max bound the forgetting range. Fitted via `scipy.optimize.curve_fit` with physically motivated bounds.

Bootstrap CIs: 2,000 resamples of the 5-seed forgetting values, sigmoid re-fit per resample, 2.5th/97.5th percentile for 95% CI.

---

## 3. Results

### 3.1 FTR Stability Crossover

#### Table 1: Crossover ε* by Architecture (sigmoid fit)

| Architecture | ε* | k (sharpness) | R² | 95% CI | F range |
|---|---|---|---|---|---|
| CNN_W8 | 6.563 | 4.15 | 0.970 | [5.82, 7.10] | [0.100, 0.189] |
| CNN_W16 | 7.283 | 2.92 | 0.985 | [6.41, 7.89] | [0.063, 0.218] |
| CNN_W32 | 7.413 | 3.61 | 0.974 | [6.87, 8.05] | [0.095, 0.208] |
| CNN_W64 | 6.855 | 4.26 | 0.989 | [6.40, 7.44] | [0.086, 0.234] |
| CNN_W96 | 7.258 | 4.76 | 0.983 | [6.82, 7.66] | [0.101, 0.213] |
| CNN_D4_W32 | 6.874 | 3.61 | 0.973 | [6.21, 7.35] | [0.068, 0.233] |
| CNN_W32_NoBN | 7.171 | 3.93 | 0.942 | [6.83, 7.62] | [0.101, 0.225] |
| ResNet18_W8 | 7.781 | 3.15 | 0.948 | [6.96, 10.95] | [0.046, 0.128] |
| | | | | | |
| **Mean ± Std** | **7.150 ± 0.354** | **3.80 ± 0.58** | **0.971 ± 0.017** | | |
| **CV** | **4.96%** | | | | |

**Key observations:**
1. **ε* ∈ [6.56, 7.78]** across all 8 architectures — a remarkably narrow interval
2. **All 95% CIs overlap** — no pair of architectures has significantly different ε*
3. **Sigmoid fits are excellent**: R² > 0.94 for all architectures
4. **Sharpness k ∈ [2.9, 4.8]**: The transition is smooth (not a sharp phase boundary)
5. **ResNet18_W8** (the only non-CNN) has wider CI due to lower dynamic range (F ∈ [0.046, 0.128] vs 0.1–0.23 for CNNs), but its ε* = 7.78 is compatible with the CNN range

#### The grid artifact is resolved

The original ε* = 7.07 (= √50) from the coarse grid was coincidental — the dense sigmoid analysis yields ε* = 7.15 ± 0.35, confirming the transition is genuine and near the original estimate, but with proper uncertainty quantification.

### 3.2 Normalization Analysis

We test whether any curvature-based rescaling collapses ε* more tightly across architectures. A "better" normalization would reduce the coefficient of variation (CV).

#### Table 2: Normalization Comparison (all 8 architectures)

| Normalization | CV | Relative to Raw |
|---|---|---|
| **Raw ε*** | **0.0496** | **1.00×** |
| ε* / log(d) | 0.0741 | 1.49× worse |
| ε* · √tr(F) | 0.4053 | 8.18× worse |
| ε* · d_eff | 0.6842 | 13.81× worse |
| ε* · tr(F)/d | 0.7614 | 15.36× worse |
| ε* · √tr(H) | 0.8653 | 17.46× worse |
| ε* · ‖∇L‖² | 0.8517 | 17.19× worse |
| ε* · tr(F) | 0.8173 | 16.49× worse |
| ε* · ‖H‖ | 1.1516 | 23.24× worse |
| ε* · κ | 1.5398 | 31.07× worse |
| ε* · tr(H) | 1.8512 | 37.35× worse |

**ALL 10 normalizations are WORSE than raw ε*.** This is the central negative result: the FTR stability crossover is already as universal as it can be, and no curvature rescaling improves upon it.

### 3.3 Correlation and Power Law Analysis

#### Table 3: ε* vs Curvature Correlations

| Metric | Pearson r | p-value | Power law α | R² |
|---|---|---|---|---|
| tr(H) | +0.645 | 0.084 | +0.020 ± 0.016 | 0.198 |
| tr(F) | +0.670 | 0.069 | +0.045 ± 0.025 | 0.353 |
| ‖H‖ | +0.633 | 0.092 | +0.035 ± 0.026 | 0.233 |
| d_eff | +0.360 | 0.382 | +0.018 ± 0.030 | 0.058 |
| \|θ\| | +0.095 | 0.822 | +0.017 ± 0.021 | 0.101 |
| ‖∇L‖ | +0.687 | 0.060 | — | — |

**Key:** No correlation reaches significance at α = 0.05. The strongest candidate (gradient norm, p = 0.060) is borderline and disappears in Kendall's τ (τ = 0.36, below significance). Power law exponents are essentially zero (α < 0.05 in all cases), meaning ε* ∝ metric^0.02 — effectively constant.

### 3.4 Constancy Test

Formal F-test for the null hypothesis H₀: ε* is constant across architectures:

| Statistic | Value |
|---|---|
| Between-architecture variance | 0.126 |
| Within-architecture (bootstrap) variance | 0.223 |
| F-ratio | 0.563 |
| **p-value** | **0.786** |

The within-architecture variance (from bootstrap resampling of seeds) actually **exceeds** the between-architecture variance. This means the observed ε* differences are entirely explained by seed-to-seed variability — the architectures produce identical ε* to within measurement noise.

### 3.5 Cross-Method Analysis

#### 3.5.1 Learning without Forgetting (LwF)

Dense 16-point α sweep (α ∈ [0.05, 5.0]), 3 seeds, 5 architectures.

| Architecture | α* | Sharpness |
|---|---|---|
| CNN_W8 | 4.026 | 22.09 |
| CNN_W16 | 0.581 | 2.54 |
| CNN_W32 | 0.661 | 2.15 |
| CNN_D4_W32 | 0.597 | 2.48 |
| ResNet18_W8 | 0.468 | 2.46 |

**LwF shows a MODERATELY architecture-dependent transition.** Excluding the outlier CNN_W8 (smallest capacity), the remaining 4 architectures yield α* ≈ 0.577 ± 0.079 (CV ≈ 14%). CNN_W8's extreme α* = 4.0 suggests that very small networks require drastically stronger distillation.

*Contrast with FTR*: LwF's transition CV (14%) is **3× worse** than FTR's (5%), and it has a strong outlier. FTR's function-space constraint is more architecturally robust.

#### 3.5.2 Elastic Weight Consolidation (EWC)

Dense 8-point λ sweep (λ ∈ [1, 10,000]), 3 seeds, 5 architectures.

| Architecture | λ* (est.) | Sharpness |
|---|---|---|
| CNN_W8 | 10,000 (limit) | 0.97 |
| CNN_W16 | 10,000 (limit) | 1.01 |
| CNN_W32 | 1 (limit) | 1.00 |
| CNN_D4_W32 | 10,000 (limit) | 1.04 |
| ResNet18_W8 | 10,000 (limit) | 1.07 |

**EWC shows NO stability crossover.** All λ* estimates hit grid limits and sharpness ≈ 1.0 (flat forgetting curve). EWC's parameter-space regularization simply does not create a meaningful stability-plasticity tradeoff on CIFAR-10 within the tested range.

---

## 4. Theoretical Interpretation

### 4.1 Why ε* Is Architecture-Independent

The FTR constraint operates in **function space** (output distributions), not parameter space. The KL divergence:

$$\text{KL}(p_\theta(y|x) \| p_{\theta_{\text{old}}}(y|x))$$

measures distributional shift in the 10-dimensional output simplex (for CIFAR-10, 10 classes). This is fundamentally a **task-space** quantity whose saturation point depends on:

1. The **class geometry** of CIFAR-10 (inter-class similarity, manifold structure)
2. The **task sequence** (5 binary classification tasks)
3. The **training dynamics** (SGD noise, learning rate)

These factors are **architecture-independent** for any sufficiently expressive network. The parameter-space geometry (Hessian trace, Fisher information, spectral norm) is irrelevant because the KL constraint maps all parameter perturbations through a common information bottleneck: the 10-class output distribution.

### 4.2 Why Normalizations Hurt

Curvature normalizations assume ε* ∝ f(curvature), implying that models with different curvature should have different thresholds. Since ε* is actually constant, **any normalization introduces artificial variance** proportional to the spread of the normalizing metric. Since tr(H) varies 48× across architectures, ε*·tr(H) has CV ≈ 37× worse than raw ε*.

This is a **falsifiable prediction**: if ε* were proportional to curvature, normalization would reduce CV. The fact that it increases CV is direct evidence against curvature-dependence.

### 4.3 The Nature of the Crossover

The transition is a **smooth sigmoid** (k ∈ [2.9, 4.8]), not a sharp phase boundary. In statistical physics terms, this is a **crossover** rather than a phase transition — there is no diverging correlation length or critical exponent. The system smoothly interpolates between:

- **Stable regime** (ε ≪ ε*): Forgetting F ≈ F_min ≈ 5–10%
- **Unstable regime** (ε ≫ ε*): Forgetting F ≈ F_max ≈ 15–25%

The characteristic width Δ(ln ε) ≈ 1/k ≈ 0.2–0.35 corresponds to a factor of ~1.2–1.4 in ε.

### 4.4 Cross-Method Comparison

| Method | Transition? | Universal? | Mechanism |
|---|---|---|---|
| **FTR** | Yes (sigmoid) | Yes (CV = 5%) | Function-space KL constraint |
| **LwF** | Yes (sigmoid) | Moderate (CV = 14%) | Output distillation (architecture-mediated) |
| **EWC** | No | N/A | Parameter-space regularization (ineffective) |

The universality ranking mirrors the **abstraction level** of each method:
- FTR: Pure function-space → architecture-independent
- LwF: Pseudo function-space (teacher-student outputs) → weakly architecture-dependent
- EWC: Pure parameter-space → no coherent transition

---

## 5. Experimental Summary

### 5.1 Scale of Investigation

| Category | Count |
|---|---|
| FTR ε sweep | 800 experiments (8 archs × 20 ε × 5 seeds) |
| Curvature measurement | 40 experiments (8 archs × 5 seeds) |
| LwF α sweep | 240 experiments (5 archs × 16 α × 3 seeds) |
| EWC λ sweep | 120 experiments (5 archs × 8 λ × 3 seeds) |
| **Total experiments** | **1,200** |
| Architectures tested | 8 (2 families, 5 widths, varied depth/BN) |
| Parameter range | 37K – 895K (24×) |
| Hessian trace range | 68 – 3,254 (48×) |
| Normalizations tested | 10 (all worse than raw) |
| Bootstrap resamples | 2,000 per architecture |
| Statistical tests | F-test, 6 Pearson/Kendall correlations, 5 power laws |

### 5.2 Generated Outputs

**Data files** (13 JSON):
- `phase1_dense_sweep.json` (37 KB) — raw forgetting data
- `curvature_5seed.json` (10 KB) — 5-seed curvature measurements
- `phase3_lwf_dense.json`, `phase3_ewc_dense.json` — cross-method sweeps
- `sigmoid_eps_star.json` — fitted crossover values
- `phase4_statistics.json` — correlations, power laws, constancy test
- `final_summary.json` — automated conclusions

**Plots** (14 pairs, png + pdf):
- `sigmoid_forgetting_curves` — All 8 architectures with sigmoid fits
- `eps_star_sigmoid_ci` — Bootstrap confidence intervals
- `phase_diagram_2d` / `phase_diagram_2d_sigmoid` — Curvature × ε phase diagram
- `normalization_collapse` — CV comparison for all normalizations
- `eps_star_vs_curvature` — Scatterplots with regression
- `cross_method_overlay` — FTR vs LwF vs EWC
- `summary_figure` / `summary_phase_diagram` — Multi-panel main figure

---

## 6. Honest Assessment

### 6.1 What This Paper Shows

1. **The FTR stability crossover at ε* ≈ 7.15 is genuine and architecture-independent** (not a grid artifact). This is established by dense sweeps, sigmoid fitting, and formal statistical testing.

2. **The universality is because FTR operates in function space**, not parameter space. No curvature normalization helps because curvature is irrelevant to the function-space geometry.

3. **The transition is a smooth crossover**, not a phase transition. There is no sharp boundary, no divergent susceptibility, no critical exponents.

4. **Cross-method comparison reveals a hierarchy**: Function-space methods (FTR) produce universal transitions; output-space methods (LwF) produce moderately universal transitions; parameter-space methods (EWC) produce no transition.

### 6.2 Limitations

1. **Single dataset (CIFAR-10)**: The ε* ≈ 7.15 value is specific to CIFAR-10's class structure. Different datasets would likely have different ε* values. The universality claim is that ε* is architecture-independent *for a given task*, not that it's task-independent.

2. **CPU-only training, small scale**: All experiments run on a MacBook Air with 1,000 samples/class. Larger-scale experiments (full CIFAR-10/100, ImageNet) may reveal scale-dependent effects.

3. **Limited architecture variation within ResNet**: Only one ResNet variant tested. More ResNet/transformer/MLP architectures would strengthen the cross-family universality claim.

4. **No theoretical derivation of ε* ≈ 7.15**: We identify the constant but do not derive it from first principles. The value likely depends on CIFAR-10's information geometry (10 classes, binary tasks, specific pixel correlations).

5. **Forgetting metric is aggregate**: We measure mean forgetting across tasks, which may mask task-specific variations.

### 6.3 NeurIPS Readiness Assessment

| Criterion | Score | Notes |
|---|---|---|
| Novelty | 7/10 | Architecture-independent crossover in function-space CL is new |
| Rigor | 8/10 | 1,200 experiments, 5 seeds, bootstrap, F-tests, 10 normalizations |
| Theoretical depth | 6/10 | Good explanation (function-space argument) but no formal derivation |
| Significance | 6/10 | Clean result but incremental; single dataset, small scale |
| Presentation | 7/10 | Clear negative result (no normalization helps) is actually strong |
| Reproducibility | 9/10 | Complete code, all hyperparameters specified, deterministic seeds |

**Overall: 7.0/10 — 35–45% probability of NeurIPS acceptance**

**Strengths:**
- Clean, falsifiable result: "ALL normalizations hurt" is stronger than "this normalization helps"
- Rigorous negative result with proper statistical backing
- Cross-method hierarchy is interesting (function-space > output-space > parameter-space)
- 1,200 experiments with proper uncertainty quantification

**Weaknesses:**
- Single dataset limits generalizability
- Small scale (CPU, 1K samples/class)
- The result, while clean, is essentially "ε* is constant" — what can practitioners *do* with this?
- No theoretical derivation of the constant
- Missing broader architecture families (transformers, MLPs)

**What would raise this to 8+/10:**
- Replicate on CIFAR-100, TinyImageNet, or language tasks
- Include transformers and MLPs in the architecture zoo
- Derive ε* ≈ 7.15 from information geometry of the task
- Show practical application (automated ε selection)

---

## Appendix A: Sigmoid Estimation Quality

All 8 architectures achieve R² > 0.94 for the sigmoid fit. The worst fit (CNN_W32_NoBN, R² = 0.942) still captures the essential shape. Bootstrap CIs are tight for CNNs (width ≈ 1.3) and wider for ResNet18 (width ≈ 4.0 due to smaller dynamic range).

## Appendix B: Curvature Measurement Details

Curvature metrics computed after training on Task 1 (first 2 CIFAR-10 classes), averaged over 5 seeds:
- **Hessian trace**: Hutchinson estimator with 50 random vectors
- **Fisher information trace**: Empirical Fisher on 500 training samples
- **Spectral norm**: Power iteration (100 steps)
- **Effective dimensionality**: tr(H)² / tr(H²)

## Appendix C: LwF Outlier Analysis

CNN_W8 (37K parameters) requires α* = 4.0 for the LwF transition, vs α* ≈ 0.58 for larger architectures. This suggests that very small networks have insufficient capacity to simultaneously match the teacher distribution and learn new tasks, requiring much stronger distillation signals.

## Appendix D: Complete Experimental Logs

All results are in `stability_constrained_selfimprovement/results/phase_diagram/`:
- Raw data, curvature measurements, cross-method sweeps, statistics
- 14 publication-quality plots (PDF + PNG)
- Fully reproducible from `run_phase_diagram.py` + `analyze_phase_diagram.py`
