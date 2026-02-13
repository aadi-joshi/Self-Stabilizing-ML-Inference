# Stability-Constrained Self-Improving Agents via Functional Trust Regions

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
scipy>=1.10
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (1 seed, reduced epochs)
python run_all.py --quick

# Full experiment pipeline (5 seeds, all experiments)
python run_all.py

# Single experiment
python run_all.py --experiment continual_cifar --seeds 42 137

# Only regenerate figures from existing results
python run_all.py --figures-only

# Run ablation studies
python run_all.py --ablations
```

## Project Structure

```
stability_constrained_selfimprovement/
├── configs/
│   ├── default.yaml          # Main experiment config
│   └── ablation.yaml         # Ablation study variations
├── models/
│   ├── resnet.py             # SmallResNet variants (44K–1.1M params)
│   ├── transformer.py        # AlgorithmicTransformer for seq2seq tasks
│   └── rl_agent.py           # PolicyNetwork + GridWorld environment
├── metrics/
│   ├── functional_drift.py   # FunctionalDrift + CKA representation drift
│   ├── constrained_optimizer.py  # Lagrangian optimizer + EpsilonScheduler + EWC
│   └── experiment_metrics.py # Logging, aggregation, statistical analysis
├── trainers/
│   └── trainer.py            # BaseTrainer with method-switching logic
├── experiments/
│   ├── exp_continual.py      # Experiment A: CIFAR-10 sequential tasks
│   ├── exp_transformer.py    # Experiment B: copy → reverse → sort
│   ├── exp_rl.py             # Experiment C: Gridworld policy learning
│   ├── ablation_runner.py    # Systematic ablation orchestration
│   └── statistical_analysis.py  # Welch's t-test, Cohen's d, LaTeX tables
├── visualization/
│   └── plotting.py           # Publication-quality figures (300 DPI, PDF)
├── utils/
│   └── common.py             # Seeds, config loading, utilities
├── paper/
│   └── paper.tex             # Full NeurIPS-format paper draft
├── run_all.py                # Master orchestration script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Concepts

### Functional Drift
Instead of constraining parameters (like EWC), we constrain the model's *behavior*:

$$D_f(θ_t, θ_0) = E_{x∼D}[||f_{θ_t}(x) - f_{θ_0}(x)||²]$$

### Lagrangian Relaxation
The constraint is enforced softly via a Lagrange multiplier λ:

$$L_total = L_task + λ · D_f(θ_t, θ_0)$$

### Dual Gradient Ascent
λ self-adjusts: increases when drift exceeds ε, decreases when within budget:

$$λ_{t+1} = max(0, λ_t + η_λ(D_f - ε))$$

## Experiments

| Experiment | Domain | Model | Tasks |
|-----------|--------|-------|-------|
| A | Continual Learning | SmallResNet (44K) | 5 CIFAR-10 binary tasks |
| B | Algorithmic Reasoning | Transformer (50K) | copy → reverse → sort |
| C | Reinforcement Learning | PolicyNetwork (MLP) | 8×8 Gridworld navigation |

## Baselines

1. **Standard Adam** — No regularization
2. **Weight Decay** — L2 parameter regularization
3. **EWC** — Elastic Weight Consolidation (Fisher-based)
4. **KL Trust Region** — KL divergence constraint (RL only)
5. **Functional Trust Region (Ours)** — Output-space drift constraint

## Citation

```bibtex
@article{functional_trust_regions_2024,
  title={Stability-Constrained Self-Improving Agents via Functional Trust Regions},
  year={2024}
}
```
