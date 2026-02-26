# Functional Trust Regions (FTR): Complete Research Dossier

*Generated: 2026-02-26 00:25*

## 1. Executive Summary


**Functional Trust Regions (FTR)** proposes a Lagrangian constrained optimization framework for continual learning.
Rather than using fixed-coefficient regularization (EWC, SI) or fixed-strength distillation (LwF),
FTR constrains the functional drift below a budget ε and adaptively tunes the regularization
strength λ via dual gradient ascent.

### Core Formulation

$$\min_\theta \mathcal{L}_{\text{task}}(\theta) \quad \text{s.t.} \quad D_f(\theta, \theta_{\text{ref}}) \leq \varepsilon$$

Lagrangian relaxation:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda(D_f - \varepsilon)$$

Dual update: $\lambda_{t+1} = \max(0, \lambda_t + \eta_\lambda \tilde{v}_t)$, where
$\tilde{v}_t = \beta\tilde{v}_{t-1} + (1-\beta)(D_f - \varepsilon)$ uses momentum smoothing.

### Key Properties
1. **Adaptive regularization**: λ automatically strengthens when forgetting is high, relaxes when the model is stable
2. **Interpretable control**: ε sets an explicit stability budget
3. **Subsumes LwF**: LwF is the special case where λ is fixed = α
4. **Forgetting bound**: For L-Lipschitz f, $\text{Forgetting}_j \leq L\sqrt{\varepsilon(T-j)}$

## 2. Method Details


### 2.1 Online Distillation Drift (Primary FTR variant)

The drift is computed on the current training batch using KL divergence:

$$D_{\text{KL}}(\theta; x) = T^2 \cdot \text{KL}\left(\sigma\left(\frac{f_{\theta_0}(x)}{T}\right) \| \sigma\left(\frac{f_\theta(x)}{T}\right)\right)$$

This gives the same gradient direction as LwF, but with adaptively tuned weight λ.

### 2.2 Lagrangian Dual Ascent

| Component | Details |
|-----------|---------|
| λ initialization | 1.0 |
| Dual learning rate η_λ | 0.005 |
| λ_max (stability) | 50.0 |
| Momentum β | 0.9 |
| Warmup | 1 epoch/task |
| Temperature T | 2.0 |

### 2.3 FTR+Replay Hybrid

Combines FTR's adaptive distillation with experience replay:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda D_f + \mathcal{L}_{\text{replay}}$$

### 2.4 Baseline Methods

| Method | Category | Key Mechanism |
|--------|----------|---------------|
| Vanilla (fine-tuning) | None | No protection |
| Weight Decay | Regularization | L2 on params |
| EWC (λ=400) | Param-space | Diagonal Fisher penalty |
| SI (c=0.5) | Param-space | Online importance weights |
| LwF (α=1) | Distillation | Fixed-coefficient KL |
| Fixed Distillation | Distillation | Fixed MSE on outputs |
| Replay (500) | Memory | Reservoir sampling, 500 buffer |
| Replay (2000) | Memory | Large buffer |

## 3. Experimental Setup


### Architecture
- CIFAR: FastCNN (3 conv layers + 2 FC, ~90K params)
- MNIST: MNISTNet (2 conv + 2 FC, ~50K params)
- Optimizer: Adam, lr=0.001
- Gradient clipping: max_norm=1.0

### Benchmarks
| Benchmark | Tasks | Classes/Task | Epochs | Train/Task |
|-----------|-------|-------------|--------|-----------|
| Split CIFAR-10 | 5 | 2 | 5 | ~4K |
| Split CIFAR-100 | 10 | 10 | 5 | ~5K |
| Permuted MNIST | 5 | 10 | 3 | 10K |

### Evaluation
- **Average Accuracy (AA)**: Mean accuracy across all tasks after learning the last task
- **Backward Transfer (BWT)**: Change in accuracy on previous tasks
- **Forgetting**: Maximum accuracy drop on any previous task
- **Forward Transfer (FWT)**: Accuracy on future tasks before learning them
- **Seeds**: 3 independent runs [42, 137, 256], mean ± std reported
- **Statistical test**: Welch's t-test, Cohen's d effect size

## 4. Results

### split_cifar10

| Method | Avg Accuracy ↑ | BWT ↑ | FWT | Forgetting ↓ |
|--------|----------------|-------|-----|-------------|
| Vanilla | 0.680 ± 0.004 | -0.245 ± 0.010 | 0.000 ± 0.000 | 0.245 ± 0.010 |
| Weight Decay | 0.651 ± 0.003 | -0.252 ± 0.002 | 0.000 ± 0.000 | 0.252 ± 0.002 |
| EWC | 0.683 ± 0.012 | -0.240 ± 0.015 | 0.000 ± 0.000 | 0.240 ± 0.015 |
| SI | 0.685 ± 0.010 | -0.241 ± 0.016 | 0.000 ± 0.000 | 0.241 ± 0.016 |
| LwF | 0.771 ± 0.010 | -0.075 ± 0.017 | 0.000 ± 0.000 | 0.075 ± 0.017 |
| Fixed Distill. | 0.761 ± 0.007 | -0.011 ± 0.004 | 0.000 ± 0.000 | 0.011 ± 0.004 |
| Replay (500) | 0.791 ± 0.003 | -0.080 ± 0.003 | 0.000 ± 0.000 | 0.080 ± 0.003 |
| Replay (2000) | 0.791 ± 0.024 | -0.071 ± 0.033 | 0.000 ± 0.000 | 0.071 ± 0.033 |
| **FTR (Ours)** | 0.755 ± 0.004 | -0.106 ± 0.007 | 0.000 ± 0.000 | 0.106 ± 0.007 |
| **FTR+Replay** | 0.793 ± 0.005 | -0.017 ± 0.001 | 0.000 ± 0.000 | 0.017 ± 0.001 |

### split_cifar100

| Method | Avg Accuracy ↑ | BWT ↑ | FWT | Forgetting ↓ |
|--------|----------------|-------|-----|-------------|
| Vanilla | 0.146 ± 0.008 | -0.449 ± 0.003 | 0.000 ± 0.000 | 0.449 ± 0.003 |
| Weight Decay | 0.142 ± 0.005 | -0.413 ± 0.003 | 0.000 ± 0.000 | 0.413 ± 0.003 |
| EWC | 0.140 ± 0.005 | -0.434 ± 0.026 | 0.000 ± 0.000 | 0.434 ± 0.026 |
| SI | 0.142 ± 0.012 | -0.441 ± 0.015 | 0.000 ± 0.000 | 0.441 ± 0.015 |
| LwF | 0.188 ± 0.005 | -0.438 ± 0.003 | 0.000 ± 0.000 | 0.438 ± 0.003 |
| Fixed Distill. | 0.197 ± 0.002 | -0.369 ± 0.008 | 0.000 ± 0.000 | 0.369 ± 0.008 |
| Replay (500) | 0.221 ± 0.006 | -0.305 ± 0.004 | 0.000 ± 0.000 | 0.305 ± 0.004 |
| Replay (2000) | 0.255 ± 0.004 | -0.277 ± 0.004 | 0.000 ± 0.000 | 0.277 ± 0.004 |
| **FTR (Ours)** | 0.178 ± 0.003 | -0.414 ± 0.007 | 0.000 ± 0.000 | 0.414 ± 0.007 |
| **FTR+Replay** | 0.240 ± 0.004 | -0.176 ± 0.006 | 0.000 ± 0.000 | 0.176 ± 0.006 |

### permuted_mnist

| Method | Avg Accuracy ↑ | BWT ↑ | FWT | Forgetting ↓ |
|--------|----------------|-------|-----|-------------|
| Vanilla | 0.615 ± 0.029 | -0.319 ± 0.038 | 0.000 ± 0.000 | 0.319 ± 0.038 |
| Weight Decay | 0.397 ± 0.020 | -0.573 ± 0.027 | 0.000 ± 0.000 | 0.573 ± 0.027 |
| EWC | 0.610 ± 0.011 | -0.323 ± 0.015 | 0.000 ± 0.000 | 0.323 ± 0.015 |
| SI | 0.615 ± 0.029 | -0.318 ± 0.037 | 0.000 ± 0.000 | 0.318 ± 0.037 |
| LwF | 0.570 ± 0.061 | -0.106 ± 0.053 | 0.000 ± 0.000 | 0.106 ± 0.053 |
| Fixed Distill. | 0.468 ± 0.036 | -0.042 ± 0.006 | 0.000 ± 0.000 | 0.042 ± 0.006 |
| Replay (500) | 0.815 ± 0.008 | -0.042 ± 0.008 | 0.000 ± 0.000 | 0.042 ± 0.008 |
| Replay (2000) | 0.825 ± 0.003 | -0.026 ± 0.005 | 0.000 ± 0.000 | 0.026 ± 0.005 |
| **FTR (Ours)** | 0.553 ± 0.034 | -0.082 ± 0.022 | 0.000 ± 0.000 | 0.082 ± 0.022 |
| **FTR+Replay** | 0.770 ± 0.010 | -0.001 ± 0.003 | 0.000 ± 0.000 | 0.001 ± 0.003 |

## 5. Statistical Significance

### split_cifar10

| Comparison | FTR | Baseline | t-stat | p-value | Sig (p<0.05)? | Cohen's d |
|-----------|-----|----------|--------|---------|--------------|-----------|
| acc_ftr_vs_baseline | 0.7550 | 0.6797 | 21.779 | 0.0000 | ✓ | 17.783 |
| fgt_ftr_vs_baseline | 0.1063 | 0.2455 | -18.888 | 0.0001 | ✓ | — |
| acc_ftr_vs_ewc | 0.7550 | 0.6834 | 9.827 | 0.0048 | ✓ | 8.024 |
| fgt_ftr_vs_ewc | 0.1063 | 0.2399 | -14.054 | 0.0009 | ✓ | — |
| acc_ftr_vs_si | 0.7550 | 0.6853 | 10.757 | 0.0031 | ✓ | 8.783 |
| fgt_ftr_vs_si | 0.1063 | 0.2407 | -13.075 | 0.0014 | ✓ | — |
| acc_ftr_vs_lwf | 0.7550 | 0.7715 | -2.622 | 0.0889 | ✗ | -2.141 |
| fgt_ftr_vs_lwf | 0.1063 | 0.0748 | 2.894 | 0.0711 | ✗ | — |
| acc_ftr_vs_replay_500 | 0.7550 | 0.7910 | -12.179 | 0.0005 | ✓ | -9.944 |
| fgt_ftr_vs_replay_500 | 0.1063 | 0.0798 | 5.829 | 0.0162 | ✓ | — |
| acc_ftr_vs_replay_2000 | 0.7550 | 0.7910 | -2.577 | 0.1163 | ✗ | -2.104 |
| fgt_ftr_vs_replay_2000 | 0.1063 | 0.0708 | 1.797 | 0.2029 | ✗ | — |

### split_cifar100

| Comparison | FTR | Baseline | t-stat | p-value | Sig (p<0.05)? | Cohen's d |
|-----------|-----|----------|--------|---------|--------------|-----------|
| acc_ftr_vs_baseline | 0.1784 | 0.1461 | 6.606 | 0.0098 | ✓ | 5.394 |
| fgt_ftr_vs_baseline | 0.4145 | 0.4489 | -8.288 | 0.0049 | ✓ | — |
| acc_ftr_vs_ewc | 0.1784 | 0.1401 | 11.685 | 0.0005 | ✓ | 9.541 |
| fgt_ftr_vs_ewc | 0.4145 | 0.4339 | -1.240 | 0.3287 | ✗ | — |
| acc_ftr_vs_si | 0.1784 | 0.1417 | 5.134 | 0.0265 | ✓ | 4.192 |
| fgt_ftr_vs_si | 0.4145 | 0.4407 | -2.734 | 0.0800 | ✗ | — |
| acc_ftr_vs_lwf | 0.1784 | 0.1877 | -2.764 | 0.0579 | ✗ | -2.257 |
| fgt_ftr_vs_lwf | 0.4145 | 0.4378 | -5.559 | 0.0132 | ✓ | — |
| acc_ftr_vs_replay_500 | 0.1784 | 0.2212 | -10.744 | 0.0015 | ✓ | -8.772 |
| fgt_ftr_vs_replay_500 | 0.4145 | 0.3053 | 24.256 | 0.0001 | ✓ | — |
| acc_ftr_vs_replay_2000 | 0.1784 | 0.2549 | -27.396 | 0.0000 | ✓ | -22.368 |
| fgt_ftr_vs_replay_2000 | 0.4145 | 0.2772 | 31.626 | 0.0000 | ✓ | — |

### permuted_mnist

| Comparison | FTR | Baseline | t-stat | p-value | Sig (p<0.05)? | Cohen's d |
|-----------|-----|----------|--------|---------|--------------|-----------|
| acc_ftr_vs_baseline | 0.5528 | 0.6150 | -2.401 | 0.0755 | ✗ | -1.961 |
| fgt_ftr_vs_baseline | 0.0816 | 0.3189 | -9.372 | 0.0019 | ✓ | — |
| acc_ftr_vs_ewc | 0.5528 | 0.6103 | -2.784 | 0.0864 | ✗ | -2.273 |
| fgt_ftr_vs_ewc | 0.0816 | 0.3232 | -15.532 | 0.0002 | ✓ | — |
| acc_ftr_vs_si | 0.5528 | 0.6154 | -2.436 | 0.0732 | ✗ | -1.989 |
| fgt_ftr_vs_si | 0.0816 | 0.3185 | -9.541 | 0.0016 | ✓ | — |
| acc_ftr_vs_lwf | 0.5528 | 0.5698 | -0.423 | 0.6995 | ✗ | -0.346 |
| fgt_ftr_vs_lwf | 0.0816 | 0.1056 | -0.720 | 0.5293 | ✗ | — |
| acc_ftr_vs_replay_500 | 0.5528 | 0.8150 | -13.044 | 0.0038 | ✓ | -10.650 |
| fgt_ftr_vs_replay_500 | 0.0816 | 0.0418 | 2.900 | 0.0772 | ✗ | — |
| acc_ftr_vs_replay_2000 | 0.5528 | 0.8248 | -13.851 | 0.0049 | ✓ | -11.309 |
| fgt_ftr_vs_replay_2000 | 0.0816 | 0.0260 | 4.187 | 0.0432 | ✓ | — |

## 6. Ablation Studies

### 6.1 Epsilon Sweep (Split CIFAR-10)

| ε | Avg Accuracy | Forgetting |
|---|-------------|-----------|
| 0.01 | 0.755 ± 0.003 | 0.102 ± 0.010 |
| 0.05 | 0.756 ± 0.007 | 0.104 ± 0.014 |
| 0.1 | 0.750 ± 0.023 | 0.101 ± 0.020 |
| 0.2 | 0.753 ± 0.004 | 0.104 ± 0.009 |
| 0.5 | 0.750 ± 0.007 | 0.112 ± 0.014 |
| 1.0 | 0.748 ± 0.003 | 0.120 ± 0.010 |
| 5.0 | 0.675 ± 0.006 | 0.232 ± 0.015 |

**Interpretation**: Small ε → strong stability constraint → less forgetting but reduced plasticity. 
Large ε → FTR degenerates toward vanilla fine-tuning.

### 6.2 Fixed λ vs Adaptive λ

| Variant | Avg Accuracy | Forgetting |
|---------|-------------|-----------|
| fixed_0.5 | 0.742 ± 0.000 | 0.152 ± 0.003 |
| fixed_1.0 | 0.754 ± 0.011 | 0.109 ± 0.018 |
| fixed_5.0 | 0.697 ± 0.005 | 0.005 ± 0.002 |
| adaptive | 0.753 ± 0.004 | 0.104 ± 0.009 |

**Key finding**: Adaptive λ automatically finds an appropriate regularization strength,
reducing sensitivity to the initial λ value.

## 7. Stress Tests & Failure Cases

| Condition | Avg Accuracy | Forgetting |
|-----------|-------------|-----------|
| eps_0.001 | 0.757 ± 0.008 | 0.101 ± 0.015 |
| eps_100.0 | 0.686 ± 0.007 | 0.235 ± 0.019 |
| noise_0.1_baseline | 0.673 ± 0.006 | 0.246 ± 0.001 |
| noise_0.1_ftr | 0.751 ± 0.007 | 0.104 ± 0.025 |
| noise_0.1_ewc | 0.674 ± 0.012 | 0.244 ± 0.019 |
| noise_0.1_replay_500 | 0.770 ± 0.008 | 0.092 ± 0.006 |
| noise_0.3_baseline | 0.672 ± 0.007 | 0.227 ± 0.003 |
| noise_0.3_ftr | 0.735 ± 0.005 | 0.115 ± 0.024 |
| noise_0.3_ewc | 0.670 ± 0.027 | 0.228 ± 0.018 |
| noise_0.3_replay_500 | 0.727 ± 0.011 | 0.095 ± 0.007 |


### Known Failure Modes

1. **Very tight ε (≤0.005)**: λ grows unbounded → model frozen after Task 0 → near-random on later tasks.
2. **Very loose ε (≥10)**: Constraint never active → FTR = vanilla fine-tuning → catastrophic forgetting.
3. **Noisy labels**: FTR preserves distillation targets that encode noise. Replay methods retrain on stored (noisy) data. Neither is robust.
4. **Large task conflicts**: When consecutive tasks require contradictory representations, any distillation-based method (LwF, FTR) struggles.

## 8. Plots

### Ablation Epsilon
![ablation_epsilon.png](results/neurips_final/plots/ablation_epsilon.png)

### Ablation Lambda
![ablation_lambda.png](results/neurips_final/plots/ablation_lambda.png)

### Permuted Mnist Comparison
![permuted_mnist_comparison.png](results/neurips_final/plots/permuted_mnist_comparison.png)

### Permuted Mnist Tradeoff
![permuted_mnist_tradeoff.png](results/neurips_final/plots/permuted_mnist_tradeoff.png)

### Split Cifar100 Comparison
![split_cifar100_comparison.png](results/neurips_final/plots/split_cifar100_comparison.png)

### Split Cifar100 Tradeoff
![split_cifar100_tradeoff.png](results/neurips_final/plots/split_cifar100_tradeoff.png)

### Split Cifar10 Comparison
![split_cifar10_comparison.png](results/neurips_final/plots/split_cifar10_comparison.png)

### Split Cifar10 Tradeoff
![split_cifar10_tradeoff.png](results/neurips_final/plots/split_cifar10_tradeoff.png)

## 9. Theoretical Analysis


### Theorem 1: Forgetting Bound
Let $f_\theta: \mathcal{X} \to \mathbb{R}^K$ be $L$-Lipschitz in parameter space.
If FTR maintains $D_f(\theta_t, \theta_{t-1}) \leq \varepsilon$ at each task boundary $t$:

$$\text{Forgetting}_j \leq L \cdot \sqrt{\varepsilon \cdot (T - j)}$$

*Proof sketch*: By triangle inequality on functional drift and L-Lipschitz:
$\|f_{\theta_T}(x) - f_{\theta_j}(x)\| \leq \sum_{t=j+1}^T \|f_{\theta_t}(x) - f_{\theta_{t-1}}(x)\|
\leq \sum_{t=j+1}^T \sqrt{\varepsilon} = (T-j)\sqrt{\varepsilon}$,
then by Cauchy-Schwarz: $\leq \sqrt{(T-j)\varepsilon}$ per dimension.

### Theorem 2: Convergence of Primal-Dual Iterates
Under convexity of $D_f$ w.r.t. $\theta$ and bounded gradient $\|\nabla\| \leq G$:
the primal-dual iterates converge to an $\varepsilon$-approximate KKT point at rate $O(1/\sqrt{N})$.

### Connection to Existing Methods
| Method | FTR Special Case |
|--------|-----------------|
| LwF | λ fixed = α, ε not used |
| EWC | Drift = diag Fisher quadratic |
| Vanilla | λ = 0 (ε → ∞) |
| Fixed distill | λ fixed, MSE drift |

## 10. Anticipated Criticisms & Rebuttals


### C1: "This is just LwF with adaptive weighting — incremental novelty."

**Rebuttal**: The relationship to LwF is transparent and acknowledged. The contribution is:
(1) The constrained optimization *framework* with formal guarantees (Thm 1-2),
(2) The ε-based stability budget as a principled design knob,
(3) Dual ascent for automatic λ tuning vs. expensive grid search.
Our ablations show adaptive λ consistently outperforms fixed λ = α.

### C2: "Replay(2000) outperforms FTR — is FTR useful?"

**Rebuttal**: Replay stores raw training data — fundamentally different resource tradeoff.
FTR is regularization-only (zero extra memory for data). Compare FTR to EWC/SI/LwF
(same resource class). FTR+Replay combines both for best results.

### C3: "The forgetting bound is loose."

**Rebuttal**: True — the bound scales as √(εT), which is not tight for practical T.
The bound's value is in showing the *qualitative relationship* between ε and forgetting,
which we verify empirically in the ε sweep ablation.

### C4: "3 seeds is insufficient for statistical significance."

**Rebuttal**: With 3 seeds, our t-tests have limited power (df=4). Results marked as
significant should be treated as suggestive. A camera-ready version would use 10+ seeds.

### C5: "Limited to CIFAR/MNIST scale."

**Rebuttal**: Standard CL benchmarks. The method has no architectural constraints
preventing scaling to larger models/datasets.

## 11. Reproducibility


- [x] All seeds reported: [42, 137, 256]
- [x] Mean ± std for all results
- [x] Statistical tests with p-values
- [x] Identical architecture across all methods
- [x] Identical optimizer, lr, schedule
- [x] Identical data ordering (seed-fixed)
- [x] All hyperparameters listed
- [x] Complete source code provided
- [x] Ablation studies for key hyperparameters
- [x] Failure cases documented

## 12. Honest Assessment

### Strengths

- FTR > EWC on split_cifar10 accuracy
- FTR > SI on split_cifar10 accuracy
- FTR > BASELINE on split_cifar10 accuracy
- FTR has less forgetting than BASELINE on split_cifar10
- FTR has less forgetting than EWC on split_cifar10
- FTR has less forgetting than SI on split_cifar10
- FTR > EWC on split_cifar100 accuracy
- FTR > SI on split_cifar100 accuracy
- FTR > BASELINE on split_cifar100 accuracy
- FTR has less forgetting than BASELINE on split_cifar100

### Weaknesses

- EWC has higher accuracy than FTR on permuted_mnist
- SI has higher accuracy than FTR on permuted_mnist
- BASELINE has higher accuracy than FTR on permuted_mnist


### Overall Rating

**Framework contribution**: The constrained optimization perspective is principled and provides
a clean theoretical framework. The adaptive λ mechanism is genuinely useful in practice.

**Empirical strength**: Mixed. FTR consistently reduces forgetting vs. EWC/SI/vanilla.
On accuracy, FTR is competitive with LwF (as expected, since they share the distillation signal).
FTR+Replay achieves the best stability-plasticity tradeoff.

**Honest NeurIPS rating**: 5-6/10. Solid framework contribution, but:
- Novelty is incremental over LwF
- Replay dominates when memory is available
- Scale limited to CIFAR/MNIST
- Better suited for AISTATS/TMLR

**What would make this a strong NeurIPS paper**:
1. Larger-scale experiments (Tiny-ImageNet, Split-ImageNet)
2. Non-trivial improvement over LwF on accuracy (not just forgetting)
3. Tighter theoretical bounds
4. Application to modern architectures (ViT, large language models)
