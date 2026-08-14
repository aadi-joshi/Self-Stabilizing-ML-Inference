#!/usr/bin/env python3
"""
Kaggle entry point. Copies the campaign code package from the input dataset
into a real Python package on sys.path, then runs the requested stage(s),
auto-sharding across all visible GPUs via subprocesses.

Edit STAGES_TO_RUN and TIME_BUDGET_HOURS_PER_STAGE below and re-push a new
kernel version for each batch of stages.
"""
import os
import shutil
import subprocess
import sys
import time

STAGES_TO_RUN = ['cross_method', 'cifar100_granularity', 'task_orderings', 'kl_ablation', 'class_incremental']
TIME_BUDGET_HOURS_PER_STAGE = 2.0  # per shard/process; kernel session cap is ~9-12h total

WORK_DIR = '/kaggle/working'
# Kaggle's dataset mount path has been observed to vary between kernel
# sessions on this account: sometimes /kaggle/input/<slug>, sometimes
# /kaggle/input/datasets/<owner>/<slug>. Rather than hardcode one, search
# for a directory containing the package's own marker file.
_CODE_SRC_CANDIDATES = [
    '/kaggle/input/ftr-campaign-code',
    '/kaggle/input/datasets/aadijoshi19/ftr-campaign-code',
]
_STATE_SRC_CANDIDATES = [
    '/kaggle/input/ftr-campaign-state',
    '/kaggle/input/datasets/aadijoshi19/ftr-campaign-state',
]


def _find_dataset_dir(candidates, marker_file):
    for c in candidates:
        if os.path.isfile(os.path.join(c, marker_file)):
            return c
    # last resort: walk /kaggle/input looking for the marker file
    for root, dirs, files in os.walk('/kaggle/input'):
        if marker_file in files:
            return root
    return None

# Fallback torch/torchvision build that still ships SASS/PTX for Pascal
# (compute capability 6.0, e.g. Tesla P100). Kaggle's stock preinstalled
# torch (observed: 2.10.0+cu128) has dropped sm_60, which fails at kernel
# LAUNCH time (not import time) with "CUDA error: no kernel image is
# available for execution on the device" -- every training step errors
# instantly, silently burning GPU-hour quota on 100%-failed runs. Detected
# via nvidia-smi BEFORE importing torch, so the reinstall (if needed)
# happens before torch's compiled extensions are loaded into the process.
FALLBACK_TORCH_INDEX = 'https://download.pytorch.org/whl/cu118'
# Oldest-first: older cu118 builds are more likely to still ship Pascal
# (sm_60) SASS/PTX. Each candidate is tried in a throwaway subprocess with a
# real CUDA kernel launch (not just import) before being accepted.
FALLBACK_CANDIDATES = [
    ('torch==2.2.2', 'torchvision==0.17.2'),
    ('torch==2.3.1', 'torchvision==0.18.1'),
    ('torch==2.4.1', 'torchvision==0.19.1'),
]

_CUDA_SELFTEST_SRC = (
    "import torch, sys\n"
    "x = torch.randn(4, 3, 8, 8, device='cuda:0')\n"
    "w = torch.nn.Conv2d(3, 4, 3).cuda()\n"
    "y = w(x); y.sum().backward()\n"
    "print('OK', torch.__version__)\n"
)


def _cuda_selftest_subprocess():
    try:
        r = subprocess.run([sys.executable, '-c', _CUDA_SELFTEST_SRC],
                            capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0 and 'OK' in r.stdout
        return ok, (r.stdout + r.stderr)
    except Exception as e:
        return False, str(e)


def ensure_gpu_compatible_torch():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            text=True, timeout=30).strip()
    except Exception as e:
        print(f"nvidia-smi query failed ({e}); assuming stock torch is fine")
        return
    if not out:
        print("no GPU reported by nvidia-smi; skipping compatibility check")
        return
    print(f"nvidia-smi GPU(s): {out}")
    caps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                caps.append(float(parts[1]))
            except ValueError:
                pass
    if not caps:
        return
    min_cap = min(caps)

    ok, msg = _cuda_selftest_subprocess()
    if ok:
        print(f"stock torch passes CUDA self-test (compute capability {min_cap}); no reinstall needed")
        return
    print(f"stock torch FAILS CUDA self-test (compute capability {min_cap}):\n{msg[-800:]}")

    for torch_pkg, tv_pkg in FALLBACK_CANDIDATES:
        print(f"trying fallback build {torch_pkg} {tv_pkg} from {FALLBACK_TORCH_INDEX} ...")
        t0 = time.time()
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
                             '--index-url', FALLBACK_TORCH_INDEX, torch_pkg, tv_pkg],
                            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  install failed ({time.time()-t0:.0f}s): {r.stderr[-500:]}")
            continue
        ok, msg = _cuda_selftest_subprocess()
        print(f"  install+selftest took {time.time()-t0:.0f}s, ok={ok}")
        if ok:
            print(f"  fallback build {torch_pkg} WORKS on this GPU")
            return
        else:
            print(f"  fallback build {torch_pkg} still fails: {msg[-500:]}")

    raise RuntimeError(
        f"No candidate torch build (stock or fallback {FALLBACK_CANDIDATES}) "
        f"passes a CUDA kernel launch on this GPU (compute capability {min_cap}). "
        f"Aborting rather than burning GPU-hour quota on guaranteed-failing runs.")


def setup_package():
    pkg_dir = os.path.join(WORK_DIR, 'campaign')
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    code_src = None
    for attempt in range(10):
        code_src = _find_dataset_dir(_CODE_SRC_CANDIDATES, 'models.py')
        if code_src:
            print(f"found code dataset at {code_src}")
            shutil.copytree(code_src, pkg_dir)
            sys.path.insert(0, WORK_DIR)
            return
        print(f"code dataset not found on attempt {attempt+1}/10 (tried {_CODE_SRC_CANDIDATES} "
              f"and a full /kaggle/input walk); listing tree and retrying in 30s...")
        for root, dirs, files in os.walk('/kaggle/input'):
            depth = root.count(os.sep) - '/kaggle/input'.count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            print(' ', root, dirs, files[:5])
        time.sleep(30)
    raise FileNotFoundError("could not locate ftr-campaign-code dataset under /kaggle/input")


def main():
    ensure_gpu_compatible_torch()  # must run before `import torch`
    setup_package()
    import torch
    n_gpu = torch.cuda.device_count()
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} n_gpu={n_gpu}")
    for i in range(n_gpu):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}, "
              f"capability={torch.cuda.get_device_capability(i)}")
    if n_gpu > 0:
        try:
            x = torch.randn(4, 3, 32, 32, device='cuda:0')
            w = torch.nn.Conv2d(3, 8, 3).cuda()
            y = w(x)
            y.sum().backward()
            print("GPU self-test (conv2d fwd+bwd) OK")
        except Exception as e:
            print(f"GPU self-test FAILED even after compatibility check: {e}")
            raise

    state_dir = os.path.join(WORK_DIR, 'state')
    os.makedirs(state_dir, exist_ok=True)
    # seed with any prior state dataset attached to this kernel (for resumed runs)
    state_src = None
    for cand in _STATE_SRC_CANDIDATES:
        if os.path.isdir(cand):
            state_src = cand
            break
    if not state_src:
        for root, dirs, files in os.walk('/kaggle/input'):
            if any(fn.endswith('__shard0of1.json') for fn in files):
                state_src = root
                break
    if state_src:
        for fn in os.listdir(state_src):
            if fn.endswith('.json'):
                shutil.copy(os.path.join(state_src, fn), os.path.join(state_dir, fn))
        print(f"Seeded state from {state_src}")
    else:
        print("no prior-state dataset found; starting fresh")

    data_root = os.path.join(WORK_DIR, 'data')
    num_shards = max(n_gpu, 1)

    for stage in STAGES_TO_RUN:
        print(f"\n{'='*70}\nSTAGE {stage} -- {num_shards} shard(s)\n{'='*70}")
        procs = []
        for shard_id in range(num_shards):
            device = f'cuda:{shard_id}' if n_gpu > 0 else 'cpu'
            cmd = [
                sys.executable, '-m', 'campaign.run',
                '--stage', stage,
                '--device', device,
                '--shard-id', str(shard_id),
                '--num-shards', str(num_shards),
                '--time-budget-hours', str(TIME_BUDGET_HOURS_PER_STAGE),
                '--state-dir', state_dir,
                '--out-dir', state_dir,
                '--data-root', data_root,
            ]
            print(f"  launching shard {shard_id}: {' '.join(cmd)}")
            log_path = os.path.join(WORK_DIR, f'{stage}_shard{shard_id}.log')
            logf = open(log_path, 'w')
            p = subprocess.Popen(cmd, cwd=WORK_DIR, stdout=logf, stderr=subprocess.STDOUT)
            procs.append((p, logf))
            if shard_id == 0:
                # let shard 0 trigger the (single, cached) CIFAR download before
                # launching concurrent shards to avoid a race on the same files
                time.sleep(60)

        for p, logf in procs:
            p.wait()
            logf.close()
        for shard_id in range(num_shards):
            log_path = os.path.join(WORK_DIR, f'{stage}_shard{shard_id}.log')
            print(f"\n--- tail of {log_path} ---")
            with open(log_path) as f:
                lines = f.readlines()
                print(''.join(lines[-30:]))

    print("\nALL STAGES DONE. State files:")
    for fn in sorted(os.listdir(state_dir)):
        print(f"  {fn}")


if __name__ == '__main__':
    main()
