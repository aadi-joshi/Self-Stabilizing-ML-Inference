#!/usr/bin/env python3
"""
Local, network-free smoke test. Monkeypatches campaign.data._load_raw with
random tensors (same shapes as real CIFAR) so the whole pipeline -- task
construction, all 30 architectures, the FTR/EWC/LwF/SI training loop, KL
directions, class-incremental mode, curvature measurement, sigmoid fit,
hierarchical model -- can be exercised end to end on CPU in seconds, without
downloading anything. This is NOT a correctness check of the science; it
only proves the code runs without crashing before spending Kaggle GPU hours.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from campaign import data as data_mod


def fake_load_raw(dataset_name, data_root):
    key = dataset_name
    if key in data_mod._CACHE:
        return data_mod._CACHE[key]
    n_classes = 10 if dataset_name == 'cifar10' else 100
    n_train, n_test = 600 * (n_classes // 10) if n_classes >= 10 else 600, 100 * (n_classes // 10) if n_classes >= 10 else 100
    n_train = max(n_train, 400)
    n_test = max(n_test, 100)
    g = torch.Generator().manual_seed(0)
    trx = torch.randn(n_train, 3, 32, 32, generator=g)
    try_ = torch.randint(0, n_classes, (n_train,), generator=g)
    tex = torch.randn(n_test, 3, 32, 32, generator=g)
    tey = torch.randint(0, n_classes, (n_test,), generator=g)
    data_mod._CACHE[key] = (trx, try_, tex, tey, n_classes)
    return data_mod._CACHE[key]


data_mod._load_raw = fake_load_raw

from campaign import models as models_mod
from campaign import engine
from campaign import analysis
from campaign import stages as stages_mod


def main():
    t0 = time.time()
    zoo = models_mod.get_architecture_zoo()
    print(f"Zoo: {len(zoo)} architectures")
    device = torch.device('cpu')

    print("\n--- task construction (task-incremental, canonical order) ---")
    tasks = data_mod.build_tasks(dataset_name='cifar10', classes_per_task=2, max_per_class=50, batch_size=32)
    assert len(tasks) == 5
    print(f"  {len(tasks)} tasks, task0 num_classes={tasks[0]['num_classes']}")

    print("\n--- task construction (class-incremental) ---")
    tasks_ci = data_mod.build_tasks(dataset_name='cifar10', classes_per_task=2, max_per_class=50,
                                     batch_size=32, class_incremental=True)
    assert tasks_ci[0]['num_classes'] == 10
    print(f"  class-incremental head size = {tasks_ci[0]['num_classes']}")

    print("\n--- task construction (permuted order, CIFAR-100 granularity) ---")
    order = data_mod.random_task_order(1001, total_classes=10)
    tasks_perm = data_mod.build_tasks(dataset_name='cifar10', classes_per_task=2, max_per_class=50,
                                       batch_size=32, task_order=order)
    tasks_c100 = data_mod.build_tasks(dataset_name='cifar100', classes_per_task=5, total_classes=20,
                                       max_per_class=20, batch_size=32)
    print(f"  permuted classes: {[t['classes'] for t in tasks_perm]}")
    print(f"  cifar100 granularity=5: {len(tasks_c100)} tasks")

    print("\n--- run_cl_experiment: every architecture, method='ftr', 1 epoch ---")
    fail = []
    for name, cfg in zoo.items():
        try:
            r = engine.run_cl_experiment(tasks, cfg['factory'], 'ftr', 42, device,
                                          epochs_per_task=1,
                                          method_cfg={'epsilon': 7.0, 'lambda_init': 1.0, 'lambda_lr': 0.005})
            assert 0 <= r['forgetting'] <= 1.01, r['forgetting']
            assert 0 <= r['new_task_accuracy'] <= 1.01
        except Exception as e:
            fail.append((name, str(e)))
            print(f"  FAIL {name}: {e}")
    if fail:
        raise SystemExit(f"{len(fail)} architectures failed: {[f[0] for f in fail]}")
    print(f"  all {len(zoo)} architectures OK")

    print("\n--- methods: ewc, si, lwf (fwd/rev/js), replay, ftr_replay ---")
    small = zoo['CNN_W8']
    for method in ['ewc', 'si', 'lwf', 'replay', 'ftr_replay']:
        r = engine.run_cl_experiment(tasks, small['factory'], method, 42, device, epochs_per_task=1,
                                      method_cfg={'epsilon': 7.0, 'ewc_lambda': 100, 'si_c': 0.5, 'lwf_alpha': 1.0})
        print(f"  {method}: forgetting={r['forgetting']:.3f} new_task_acc={r['new_task_accuracy']:.3f}")
    for direction in ['forward', 'reverse', 'js']:
        r = engine.run_cl_experiment(tasks, small['factory'], 'ftr', 42, device, epochs_per_task=1,
                                      method_cfg={'epsilon': 7.0}, kl_direction=direction)
        print(f"  kl_direction={direction}: forgetting={r['forgetting']:.3f}")

    print("\n--- class-incremental run ---")
    r = engine.run_cl_experiment(tasks_ci, small['factory'], 'ftr', 42, device, epochs_per_task=1,
                                  method_cfg={'epsilon': 7.0})
    print(f"  class-incremental forgetting={r['forgetting']:.3f}")

    print("\n--- trajectory logging ---")
    r = engine.run_cl_experiment(tasks, small['factory'], 'ftr', 42, device, epochs_per_task=1,
                                  method_cfg={'epsilon': 7.0}, log_trajectory=True)
    assert 'trajectory' in r and len(r['trajectory']['lambda']) > 0
    print(f"  trajectory length={len(r['trajectory']['lambda'])}")

    print("\n--- curvature measurement ---")
    c = engine.measure_intrinsic_curvature(small['factory'], tasks, 42, device, epochs=1, n_hutch=2, n_fisher_batches=2)
    print(f"  {c}")

    print("\n--- analysis: sigmoid fit + bootstrap + hierarchical model ---")
    import numpy as np
    eps_vals = [1, 3, 5, 7, 9, 12, 20]
    fg_by_seed = {str(e): [0.05 + 0.15 / (1 + np.exp(-1.2 * (e - 7))) + 0.01 * s for s in range(3)] for e in eps_vals}
    fg_means = [float(np.mean(fg_by_seed[str(e)])) for e in eps_vals]
    es, k, fmin, fmax, r2 = analysis.sigmoid_fit_eps_star(eps_vals, fg_means)
    print(f"  sigmoid fit: eps*={es:.2f} k={k:.2f} R2={r2:.3f}")
    bm, bs, lo, hi = analysis.bootstrap_eps_star_sigmoid(eps_vals, fg_by_seed, n_bootstrap=100)
    print(f"  bootstrap: {bm:.2f} +/- {bs:.2f}, CI [{lo:.2f},{hi:.2f}]")

    fake_eps_star = {f'arch{i}': 7.0 + 0.1 * i for i in range(8)}
    fake_std = {f'arch{i}': 0.3 for i in range(8)}
    hb = analysis.hierarchical_partial_pooling(fake_eps_star, fake_std, n_mcmc=2000, burn=500)
    print(f"  hierarchical: mu={hb['mu_posterior_mean']:.2f} tau={hb['tau_posterior_mean']:.3f} "
          f"ICC={hb['icc_posterior_mean']:.3f}")

    loo = analysis.leave_one_out(fake_eps_star, fake_std)
    print(f"  leave-one-out keys: {len(loo)}")

    fake_curv = {f'arch{i}': {'hessian_trace': 100.0 + 10 * i, 'fisher_trace': 1.0, 'spectral_norm': 50.0,
                               'd_eff': 5.0, 'n_params': 1000 * (i + 1), 'gradient_norm': 2.0} for i in range(8)}
    cp = analysis.correlation_power(fake_eps_star, fake_curv)
    print(f"  correlation_power: min detectable r at n=8 = {cp['min_detectable_r_at_80pct_power']:.3f}")

    fsig = {f'CNN_W{w}': {'sigmoid_k': 2.0 + 0.1 * i, 'eps_star_sigmoid': 7.0} for i, w in
            enumerate([8, 16, 24, 32, 48, 64, 96, 128])}
    fss = analysis.finite_size_scaling(fsig, [f'CNN_W{w}' for w in [8, 16, 24, 32, 48, 64, 96, 128]],
                                        [8, 16, 24, 32, 48, 64, 96, 128])
    print(f"  finite_size_scaling alpha={fss['power_law_alpha']:.3f} R2={fss['r_squared']:.3f}")

    print("\n--- stage item builders (counts only) ---")
    for name, (build_fn, run_fn) in stages_mod.STAGE_REGISTRY.items():
        items = build_fn(zoo)
        print(f"  {name}: {len(items)} work items")

    print(f"\nALL SMOKE TESTS PASSED in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
