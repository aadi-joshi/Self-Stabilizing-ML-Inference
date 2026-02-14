#!/bin/bash
# Run FTR experiments in background
cd /Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/stability_constrained_selfimprovement
PYTHON=/Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/.venv/bin/python
SCRIPT=/Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/stability_constrained_selfimprovement/run_lean.py
LOG=/Users/kavyabhand/Desktop/Work/Self-Stabilizing-ML-Inference-System/stability_constrained_selfimprovement/experiment_log.txt

echo "Starting experiments at $(date)" > "$LOG"
$PYTHON -u "$SCRIPT" >> "$LOG" 2>&1
echo "Finished at $(date)" >> "$LOG"
