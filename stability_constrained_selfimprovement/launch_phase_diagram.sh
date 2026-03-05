#!/bin/bash
cd /Users/kavyabhand/Desktop/SSMLIS-DEV/stability_constrained_selfimprovement
python3 run_phase_diagram.py > phase_diagram.log 2>&1 &
echo "PID=$!"
