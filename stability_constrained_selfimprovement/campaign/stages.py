"""
Experiment stage definitions for the FTR campaign.

Each stage yields a list of "work items" (dicts describing one training run)
and knows how to execute a single item. `run.py` handles checkpointing,
sharding, and time-budget cutoffs generically across all stages.

Stages:
  diagnostic        -- optimizer-schedule sensitivity of eps* (decisive test:
                        artifact of dual-ascent schedule vs task-structure
                        invariant). MUST run first; determines paper framing.
  curvature         -- Hessian/Fisher/spectral/d_eff for the full 30-arch zoo
  dense_sweep       -- main FTR eps sweep, full zoo, CIFAR-10, task-incremental
  cross_method      -- EWC/LwF/SI sweeps on the same architecture subset
  cifar100_granularity -- fixed 20-class pool, classes/task in {2,4,5,10}
                        (tests whether eps* shifts with task granularity --
                        the theory-testing experiment from NEXT.md Sec 7)
  task_orderings    -- 3 random class-to-task permutations, subset of archs
  kl_ablation       -- forward vs reverse vs Jensen-Shannon KL direction
  class_incremental -- shared single head, no oracle task ID, width family
"""
import itertools
from collections import OrderedDict

from . import data as data_mod
from . import engine

DEFAULT_EPS_GRID = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 25.0, 50.0]
DIAG_EPS_GRID = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 20.0]
SEEDS_MAIN = [42, 137, 256, 7, 2024]
SEEDS_ANCHOR = [42, 137, 256, 7, 2024, 31, 99, 555, 777, 8675309]  # 10 seeds for anchor archs
SEEDS_SUB = [42, 137, 256]

FTR_BASE_CFG = {'epsilon': 7.0, 'lambda_init': 1.0, 'lambda_lr': 0.005, 'lambda_max': 50.0,
                'lambda_momentum': 0.9, 'temperature': 2.0, 'warmup_epochs': 1}

# Architecture subset used for the cross-method / task-ordering / KL-ablation
# stages, chosen to span every family in models.get_architecture_zoo at
# roughly matched parameter scale (keeps compute bounded while still fixing
# NEXT.md 4.5's "LwF/EWC tested on a smaller, different subset" complaint --
# this subset is now used identically across ALL stages, including FTR).
SUBSET_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_D4_W32', 'CNN_W32_NoBN',
                'ResNet18_W8', 'ResNet18_W16', 'ResNetLite_W16', 'ResNetLite_W32',
                'ResNetLite_W8_NoBN', 'MLP_H128', 'ViT_Tiny', 'Mixer_Tiny', 'CNN_W96']

ANCHOR_ARCHS = ['CNN_W16', 'CNN_W32']  # used for the diagnostic + KL ablation
WIDTH_FAMILY = ['CNN_W8', 'CNN_W16', 'CNN_W24', 'CNN_W32', 'CNN_W48', 'CNN_W64', 'CNN_W96', 'CNN_W128']
WIDTHS = [8, 16, 24, 32, 48, 64, 96, 128]


def _cache_tasks(cache, key, **kwargs):
    if key not in cache:
        cache[key] = data_mod.build_tasks(**kwargs)
    return cache[key]


# ======================================================================
# STAGE: diagnostic (optimizer-schedule sensitivity)
# ======================================================================
DIAG_CONDITIONS = OrderedDict([
    ('baseline', {}),
    ('eta_lambda_x5', {'lambda_lr': 0.025}),
    ('eta_lambda_div5', {'lambda_lr': 0.001}),
    ('lambda_init_x5', {'lambda_init': 5.0}),
    ('lambda_init_div5', {'lambda_init': 0.2}),
    ('lambda_max_low', {'lambda_max': 10.0}),
    ('lambda_max_high', {'lambda_max': 200.0}),
    ('epochs_x2', {'_epochs_override': 8}),
    ('epochs_div2', {'_epochs_override': 2}),
    ('momentum_low', {'lambda_momentum': 0.5}),
    ('temperature_x2', {'temperature': 4.0}),
    ('temperature_div2', {'temperature': 1.0}),
])


def build_diagnostic_items(zoo):
    items = []
    for arch in ANCHOR_ARCHS:
        for cond_name, overrides in DIAG_CONDITIONS.items():
            for eps in DIAG_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({
                        'stage': 'diagnostic', 'arch': arch, 'cond': cond_name,
                        'eps': eps, 'seed': seed, 'overrides': overrides,
                    })
    return items


def run_diagnostic_item(item, zoo, tasks_cache, device, log_trajectory=False):
    arch = item['arch']
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg.update(item['overrides'])
    cfg['epsilon'] = item['eps']
    epochs = cfg.pop('_epochs_override', zoo[arch]['epochs'])
    r = engine.run_cl_experiment(tasks, zoo[arch]['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=epochs, method_cfg=cfg, log_trajectory=log_trajectory)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: curvature (full zoo)
# ======================================================================
def build_curvature_items(zoo, seeds=SEEDS_MAIN):
    return [{'stage': 'curvature', 'arch': a, 'seed': s} for a in zoo for s in seeds]


def run_curvature_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    arch_cfg = zoo[item['arch']]
    return engine.measure_intrinsic_curvature(arch_cfg['factory'], tasks, item['seed'], device,
                                               epochs=arch_cfg['epochs'], n_hutch=10, n_fisher_batches=10)


# ======================================================================
# STAGE: dense_sweep (main FTR result, full zoo, CIFAR-10)
# ======================================================================
def build_dense_sweep_items(zoo):
    items = []
    for arch in zoo:
        seeds = SEEDS_ANCHOR if arch in ANCHOR_ARCHS else SEEDS_MAIN
        for eps in DEFAULT_EPS_GRID:
            for seed in seeds:
                items.append({'stage': 'dense_sweep', 'arch': arch, 'eps': eps, 'seed': seed})
    return items


def run_dense_sweep_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: cross_method (EWC / LwF / SI on the same subset as FTR)
# ======================================================================
METHOD_GRIDS = {
    'ewc': {'param': 'ewc_lambda', 'values': [1, 10, 50, 100, 500, 1000, 5000, 10000]},
    'lwf': {'param': 'lwf_alpha', 'values': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0, 2.0, 5.0]},
    'si': {'param': 'si_c', 'values': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
}


def build_cross_method_items(zoo):
    items = []
    for method, grid in METHOD_GRIDS.items():
        for arch in SUBSET_ARCHS:
            for val in grid['values']:
                for seed in SEEDS_SUB:
                    items.append({'stage': 'cross_method', 'method': method, 'arch': arch,
                                  'hyper_value': val, 'seed': seed})
    return items


def run_cross_method_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    method = item['method']
    param = METHOD_GRIDS[method]['param']
    cfg = {param: item['hyper_value'], 'temperature': 2.0}
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], method, item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: cifar100_granularity (fixed 20-class pool, vary classes/task)
# ======================================================================
GRANULARITY_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_W64', 'CNN_D4_W32',
                      'ResNet18_W8', 'ResNet18_W16', 'ResNetLite_W16', 'ViT_Tiny', 'Mixer_Tiny']
GRANULARITY_EPS_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0]
CLASSES_PER_TASK_OPTS = [2, 4, 5, 10]


def build_cifar100_granularity_items(zoo):
    items = []
    for arch in GRANULARITY_ARCHS:
        for cpt in CLASSES_PER_TASK_OPTS:
            for eps in GRANULARITY_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({'stage': 'cifar100_granularity', 'arch': arch, 'cpt': cpt,
                                  'eps': eps, 'seed': seed})
    return items


def run_cifar100_granularity_item(item, zoo, tasks_cache, device):
    cpt = item['cpt']
    tasks = _cache_tasks(tasks_cache, f'cifar100_g{cpt}', dataset_name='cifar100',
                          classes_per_task=cpt, total_classes=20, max_per_class=400, batch_size=128)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: task_orderings (>=3 random class-to-task permutations)
# ======================================================================
ORDERING_ARCHS = ['CNN_W8', 'CNN_W16', 'CNN_W32', 'CNN_D4_W32', 'ResNet18_W8', 'ViT_Tiny']
ORDERING_SEEDS_FOR_PERM = [1001, 1002, 1003]  # defines 3 permutations (order 0 = canonical, already in dense_sweep)


def build_task_orderings_items(zoo):
    items = []
    for perm_seed in ORDERING_SEEDS_FOR_PERM:
        for arch in ORDERING_ARCHS:
            for eps in DEFAULT_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({'stage': 'task_orderings', 'arch': arch, 'perm_seed': perm_seed,
                                  'eps': eps, 'seed': seed})
    return items


def run_task_orderings_item(item, zoo, tasks_cache, device):
    perm_seed = item['perm_seed']
    key = f'cifar10_perm{perm_seed}'
    if key not in tasks_cache:
        order = data_mod.random_task_order(perm_seed, total_classes=10)
        tasks_cache[key] = data_mod.build_tasks(dataset_name='cifar10', classes_per_task=2,
                                                 max_per_class=1000, batch_size=256, task_order=order)
    tasks = tasks_cache[key]
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: kl_ablation (forward / reverse / JS KL direction)
# ======================================================================
def build_kl_ablation_items(zoo):
    items = []
    for arch in ANCHOR_ARCHS:
        for direction in ['forward', 'reverse', 'js']:
            for eps in DEFAULT_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({'stage': 'kl_ablation', 'arch': arch, 'direction': direction,
                                  'eps': eps, 'seed': seed})
    return items


def run_kl_ablation_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg,
                                  kl_direction=item['direction'])
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: class_incremental (shared head, no oracle task ID, width family)
# ======================================================================
def build_class_incremental_items(zoo):
    items = []
    for arch in WIDTH_FAMILY[:5]:  # W8..W48, keep compute bounded (10-way head is costlier)
        for eps in DEFAULT_EPS_GRID:
            for seed in SEEDS_SUB:
                items.append({'stage': 'class_incremental', 'arch': arch, 'eps': eps, 'seed': seed})
    return items


def run_class_incremental_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_class_incr', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256,
                          class_incremental=True)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: diagnostic_wide (pin down the saturated conditions from `diagnostic`
# with a grid wide enough to actually locate their crossover instead of
# extrapolating past [1,20])
# ======================================================================
WIDE_EPS_GRID = [0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
SATURATED_CONDITIONS = OrderedDict([
    ('baseline', {}),  # re-run with the wide grid too, as a consistency check
    ('eta_lambda_div5', {'lambda_lr': 0.001}),
    ('lambda_init_x5', {'lambda_init': 5.0}),
    ('lambda_init_div5', {'lambda_init': 0.2}),
    ('epochs_div2', {'_epochs_override': 2}),
])


def build_diagnostic_wide_items(zoo):
    items = []
    for arch in ANCHOR_ARCHS:
        for cond_name, overrides in SATURATED_CONDITIONS.items():
            for eps in WIDE_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({
                        'stage': 'diagnostic_wide', 'arch': arch, 'cond': cond_name,
                        'eps': eps, 'seed': seed, 'overrides': overrides,
                    })
    return items


def run_diagnostic_wide_item(item, zoo, tasks_cache, device):
    return run_diagnostic_item(item, zoo, tasks_cache, device)


# ======================================================================
# STAGE: epoch_matched_control -- ViT/Mixer use epochs=5 in the zoo (vs 4
# for every other family), which the diagnostic mechanism (Sec. 5 of the
# paper) predicts shifts their eps* down by roughly lambda_init/(eta_lambda)
# * (1/24 - 1/32) independent of any real family-level effect. This stage
# re-runs ViT_Tiny and Mixer_Tiny with epochs forced to 4, matching the CNN
# family, to separate the genuine architecture-family effect from this
# self-inflicted schedule confound.
# ======================================================================
EPOCH_MATCHED_ARCHS = ['ViT_Tiny', 'Mixer_Tiny', 'ViT_Small', 'Mixer_Small']


def build_epoch_matched_control_items(zoo):
    items = []
    for arch in EPOCH_MATCHED_ARCHS:
        for eps in DEFAULT_EPS_GRID:
            for seed in SEEDS_MAIN:
                items.append({'stage': 'epoch_matched_control', 'arch': arch, 'eps': eps, 'seed': seed})
    return items


def run_epoch_matched_control_item(item, zoo, tasks_cache, device):
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=4, method_cfg=cfg)  # forced 4, not arch_cfg['epochs']
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: diagnostic_families -- extends the schedule-sensitivity diagnostic
# (previously CNN_W16/CNN_W32 only) to one anchor architecture from each of
# the four remaining families, so the confound isn't claimed on n=2, both
# CNNs. Core conditions only (the clean, high-R^2 ones from `diagnostic`,
# skipping the inert lambda_max/temperature axes to keep cost bounded).
# All archs run at epochs_per_task=4 (the corrected, matched schedule) so
# results are directly comparable across families without a second confound.
# ======================================================================
FAMILY_ANCHOR_ARCHS = ['ResNet18_W16', 'MLP_H128', 'ViT_Tiny', 'Mixer_Tiny']
CORE_DIAG_CONDITIONS = OrderedDict([
    ('baseline', {}),
    ('eta_lambda_x5', {'lambda_lr': 0.025}),
    ('epochs_x2', {'_epochs_override': 8}),
    ('lambda_init_x5', {'lambda_init': 5.0}),
    ('lambda_init_div5', {'lambda_init': 0.2}),
])


def build_diagnostic_families_items(zoo):
    items = []
    for arch in FAMILY_ANCHOR_ARCHS:
        for cond_name, overrides in CORE_DIAG_CONDITIONS.items():
            for eps in DIAG_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({
                        'stage': 'diagnostic_families', 'arch': arch, 'cond': cond_name,
                        'eps': eps, 'seed': seed, 'overrides': overrides,
                    })
    return items


def run_diagnostic_families_item(item, zoo, tasks_cache, device):
    arch = item['arch']
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg.update(item['overrides'])
    cfg['epsilon'] = item['eps']
    epochs = cfg.pop('_epochs_override', 4)  # matched schedule: 4, not zoo[arch]['epochs']
    r = engine.run_cl_experiment(tasks, zoo[arch]['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=epochs, method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: s_invariance -- tests whether Eq. 9's schedule term
# S = lambda_init / (eta_lambda * N) is the invariant quantity, not the
# three raw hyperparameters individually. Baseline S = 1.0/(0.005*24) =
# 8.333 (N=24 constrained steps at epochs=4). Four conditions hold S fixed
# at 8.333 while moving eta_lambda, lambda_init, and N individually or in
# combination by up to 5x; one condition ('mismatched_control') changes
# eta_lambda alone (identical to the original diagnostic's eta_lambda_x5)
# as a within-experiment negative control that should NOT preserve eps*.
# If eps* stays put across the four S-matched conditions but shifts under
# the mismatched control, S -- not the raw triple -- is the operative
# invariant, converting the Sec. 5 confound into a reportable protocol.
# ======================================================================
S_INVARIANCE_ARCHS = ['CNN_W16', 'CNN_W32', 'ResNet18_W16', 'MLP_H128', 'ViT_Tiny']
S_INVARIANCE_CONDITIONS = OrderedDict([
    ('baseline', {}),                                                          # S = 8.333, N=24
    ('s_matched_scale_up', {'lambda_lr': 0.025, 'lambda_init': 5.0}),          # S = 8.333, N=24
    ('s_matched_scale_down', {'lambda_lr': 0.001, 'lambda_init': 0.2}),        # S = 8.333, N=24
    ('s_matched_via_steps_a', {'lambda_lr': 0.0025, '_epochs_override': 7}),   # S = 8.333, N=48
    ('s_matched_via_steps_b', {'lambda_init': 2.0, '_epochs_override': 7}),    # S = 8.333, N=48
    ('mismatched_control', {'lambda_lr': 0.025}),                              # S = 1.667, N=24 (negative control)
])


def build_s_invariance_items(zoo):
    items = []
    for arch in S_INVARIANCE_ARCHS:
        for cond_name, overrides in S_INVARIANCE_CONDITIONS.items():
            for eps in DIAG_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({
                        'stage': 's_invariance', 'arch': arch, 'cond': cond_name,
                        'eps': eps, 'seed': seed, 'overrides': overrides,
                    })
    return items


def run_s_invariance_item(item, zoo, tasks_cache, device):
    arch = item['arch']
    tasks = _cache_tasks(tasks_cache, 'cifar10_default', dataset_name='cifar10',
                          classes_per_task=2, max_per_class=1000, batch_size=256)
    cfg = dict(FTR_BASE_CFG)
    cfg.update(item['overrides'])
    cfg['epsilon'] = item['eps']
    epochs = cfg.pop('_epochs_override', 4)
    r = engine.run_cl_experiment(tasks, zoo[arch]['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=epochs, method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: cifar100_granularity_v2 -- fixes the confound in
# `cifar100_granularity`: that stage held max_per_class fixed, so
# samples/task = cpt * max_per_class grew with cpt, changing N (steps/task)
# along with task granularity. This version holds samples/task fixed at
# 800 (= the original cpt=2 cell's sample count) by setting
# max_per_class = 800 // cpt, isolating classes/task as the only thing that
# varies.
# ======================================================================
GRANULARITY_V2_SAMPLES_PER_TASK = 800


def build_cifar100_granularity_v2_items(zoo):
    items = []
    for arch in GRANULARITY_ARCHS:
        for cpt in CLASSES_PER_TASK_OPTS:
            for eps in GRANULARITY_EPS_GRID:
                for seed in SEEDS_SUB:
                    items.append({'stage': 'cifar100_granularity_v2', 'arch': arch, 'cpt': cpt,
                                  'eps': eps, 'seed': seed})
    return items


def run_cifar100_granularity_v2_item(item, zoo, tasks_cache, device):
    cpt = item['cpt']
    max_per_class = max(GRANULARITY_V2_SAMPLES_PER_TASK // cpt, 1)
    tasks = _cache_tasks(tasks_cache, f'cifar100_gv2_{cpt}', dataset_name='cifar100',
                          classes_per_task=cpt, total_classes=20, max_per_class=max_per_class, batch_size=128)
    cfg = dict(FTR_BASE_CFG)
    cfg['epsilon'] = item['eps']
    arch_cfg = zoo[item['arch']]
    r = engine.run_cl_experiment(tasks, arch_cfg['factory'], 'ftr', item['seed'], device,
                                  epochs_per_task=arch_cfg['epochs'], method_cfg=cfg)
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: crossover_wide -- relocates the two dense_sweep architectures
# whose sigmoid fit pinned against the curve_fit upper bound
# (log(eps_grid_max)+1, i.e. eps*=50*e=135.91 for both ResNetLite_W8_NoBN
# and ResNet18_W16) on a grid extending to eps=200, the same fix already
# applied to the diagnostic's saturated conditions.
# ======================================================================
CROSSOVER_WIDE_ARCHS = ['ResNetLite_W8_NoBN', 'ResNet18_W16']
CROSSOVER_WIDE_EPS_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def build_crossover_wide_items(zoo):
    items = []
    for arch in CROSSOVER_WIDE_ARCHS:
        for eps in CROSSOVER_WIDE_EPS_GRID:
            for seed in SEEDS_MAIN:
                items.append({'stage': 'crossover_wide', 'arch': arch, 'eps': eps, 'seed': seed})
    return items


def run_crossover_wide_item(item, zoo, tasks_cache, device):
    return run_dense_sweep_item(item, zoo, tasks_cache, device)


# ======================================================================
# STAGE: pretrained_modern -- stretch experiment: pretrained ViT-B/16 +
# LoRA on Split CIFAR-100 (10 tasks x 10 classes), comparing FTR/LwF/EWC/
# vanilla fine-tuning in a setting closer to how continual learning is
# actually practiced today. Standalone model/data pipeline (see
# pretrained_experiment.py); only lightly touches the shared checkpoint
# format (arch/eps fields repurposed as method/hyperparameter labels).
# ======================================================================
PRETRAINED_METHOD_CONFIGS = OrderedDict([
    ('finetune', {}),
    ('ewc', {'ewc_lambda': 1000.0}),
    ('lwf', {'lwf_alpha': 0.7}),
    ('ftr', {'epsilon': 5.0, 'lambda_init': 1.0, 'lambda_lr': 0.005}),
])
PRETRAINED_SEEDS = [42, 137, 256]


def build_pretrained_modern_items(zoo):
    items = []
    for method, cfg in PRETRAINED_METHOD_CONFIGS.items():
        for seed in PRETRAINED_SEEDS:
            items.append({'stage': 'pretrained_modern', 'method': method, 'seed': seed, 'cfg': cfg})
    return items


def run_pretrained_modern_item(item, zoo, tasks_cache, device):
    from . import pretrained_experiment as pt
    r = pt.run_pretrained_experiment(item['method'], item['seed'], device,
                                      n_tasks=10, epochs_per_task=2, method_cfg=item['cfg'])
    r.pop('acc_matrix', None)
    return r


# ======================================================================
# STAGE: s_invariance_wide -- relocates the four (architecture, condition)
# cells from `s_invariance` that pinned against the curve_fit search bound
# on the narrow DIAG_EPS_GRID (max 20), the same 20*e=54.37 artifact seen
# in `crossover_wide`'s two cells, using a grid extending to 200.
# ======================================================================
S_INVARIANCE_WIDE_CELLS = [
    ('ResNet18_W16', 'baseline'),
    ('MLP_H128', 'baseline'),
    ('ViT_Tiny', 'baseline'),
    ('ViT_Tiny', 's_matched_scale_up'),
]
S_INVARIANCE_WIDE_EPS_GRID = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def build_s_invariance_wide_items(zoo):
    items = []
    for arch, cond_name in S_INVARIANCE_WIDE_CELLS:
        overrides = S_INVARIANCE_CONDITIONS[cond_name]
        for eps in S_INVARIANCE_WIDE_EPS_GRID:
            for seed in SEEDS_SUB:
                items.append({
                    'stage': 's_invariance_wide', 'arch': arch, 'cond': cond_name,
                    'eps': eps, 'seed': seed, 'overrides': overrides,
                })
    return items


def run_s_invariance_wide_item(item, zoo, tasks_cache, device):
    return run_s_invariance_item(item, zoo, tasks_cache, device)


STAGE_REGISTRY = {
    'diagnostic': (build_diagnostic_items, run_diagnostic_item),
    'diagnostic_wide': (build_diagnostic_wide_items, run_diagnostic_wide_item),
    'epoch_matched_control': (build_epoch_matched_control_items, run_epoch_matched_control_item),
    'curvature': (build_curvature_items, run_curvature_item),
    'dense_sweep': (build_dense_sweep_items, run_dense_sweep_item),
    'cross_method': (build_cross_method_items, run_cross_method_item),
    'cifar100_granularity': (build_cifar100_granularity_items, run_cifar100_granularity_item),
    'task_orderings': (build_task_orderings_items, run_task_orderings_item),
    'kl_ablation': (build_kl_ablation_items, run_kl_ablation_item),
    'class_incremental': (build_class_incremental_items, run_class_incremental_item),
    'diagnostic_families': (build_diagnostic_families_items, run_diagnostic_families_item),
    's_invariance': (build_s_invariance_items, run_s_invariance_item),
    'cifar100_granularity_v2': (build_cifar100_granularity_v2_items, run_cifar100_granularity_v2_item),
    'crossover_wide': (build_crossover_wide_items, run_crossover_wide_item),
    'pretrained_modern': (build_pretrained_modern_items, run_pretrained_modern_item),
    's_invariance_wide': (build_s_invariance_wide_items, run_s_invariance_wide_item),
}
