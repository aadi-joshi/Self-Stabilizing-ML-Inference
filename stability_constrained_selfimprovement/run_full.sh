#!/bin/bash
cd /Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/stability_constrained_selfimprovement
nohup /Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/.venv/bin/python -u run_full_experiment.py \
    --benchmarks split_cifar10 \
    --methods baseline ewc si lwf replay functional_trust ftr_replay \
    --seeds 42 137 256 \
    --epochs 15 \
    > experiment_log.txt 2>&1 &
echo "PID: $!"
echo "Monitor with: tail -f experiment_log.txt"
