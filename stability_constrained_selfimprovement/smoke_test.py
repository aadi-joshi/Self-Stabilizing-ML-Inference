#!/usr/bin/env python3
"""Quick smoke test for run_neurips_breakthrough.py"""
import sys, time
sys.path.insert(0, '.')
from run_neurips_breakthrough import *

set_seed(42)
tasks = load_cifar10_split(5, 256, 200)
zoo = get_architecture_zoo()

# Test FTR
t0 = time.time()
r = run_cl_experiment(tasks, zoo['CNN_W8']['factory'], 'ftr', 42, DEVICE,
                      epochs_per_task=2, method_cfg={'epsilon': 0.5})
print(f"FTR: AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.1f}s)")

# Test EWC
t0 = time.time()
r = run_cl_experiment(tasks, zoo['CNN_W8']['factory'], 'ewc', 42, DEVICE,
                      epochs_per_task=2, method_cfg={'ewc_lambda': 100})
print(f"EWC: AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.1f}s)")

# Test SI
t0 = time.time()
r = run_cl_experiment(tasks, zoo['CNN_W8']['factory'], 'si', 42, DEVICE,
                      epochs_per_task=2, method_cfg={'si_c': 1.0})
print(f"SI: AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.1f}s)")

# Test LwF
t0 = time.time()
r = run_cl_experiment(tasks, zoo['CNN_W8']['factory'], 'lwf', 42, DEVICE,
                      epochs_per_task=2, method_cfg={'lwf_alpha': 1.0})
print(f"LwF: AA={r['average_accuracy']:.3f} F={r['forgetting']:.3f} ({time.time()-t0:.1f}s)")

# Test curvature
t0 = time.time()
c = measure_intrinsic_curvature(zoo['CNN_W8']['factory'], tasks, 42, DEVICE,
                                 epochs=2, n_hutch=5, n_fisher_batches=3)
print(f"Curvature: ht={c['hessian_trace']:.1f} ft={c['fisher_trace']:.2f} sn={c['spectral_norm']:.2f} deff={c['d_eff']:.0f} ({time.time()-t0:.1f}s)")
print("ALL SMOKE TESTS PASSED")
