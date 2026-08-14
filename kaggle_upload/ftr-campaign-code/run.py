#!/usr/bin/env python3
"""
CLI runner for the FTR campaign. Designed to run standalone on Kaggle
(GPU, internet enabled for the first CIFAR download) with graceful
time-budget cutoffs and full resumability via flat JSON checkpoints.

Usage:
  python -m campaign.run --stage diagnostic --device cuda:0 \
      --shard-id 0 --num-shards 2 --time-budget-hours 3.0 \
      --state-dir ./state --out-dir ./state

Each (stage, shard) pair writes to {out_dir}/{stage}__shard{shard_id}of{num_shards}.json
Run `python -m campaign.merge` afterwards (or after downloading kernel
output) to combine all shard files per stage into one JSON per stage for
analysis.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import models as models_mod
from campaign import stages as stages_mod


def resolve_device(spec):
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)


def item_key(item):
    return json.dumps(item, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(stages_mod.STAGE_REGISTRY.keys()) + ['all'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--time-budget-hours', type=float, default=100.0)
    ap.add_argument('--state-dir', default='./state')
    ap.add_argument('--out-dir', default='./state')
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--save-every', type=int, default=1, help='checkpoint every N completed items')
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}, torch {torch.__version__}, cuda available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device count: {torch.cuda.device_count()}")

    os.makedirs(args.out_dir, exist_ok=True)
    zoo = models_mod.get_architecture_zoo()
    print(f"Architecture zoo: {len(zoo)} architectures")

    stages_to_run = list(stages_mod.STAGE_REGISTRY.keys()) if args.stage == 'all' else [args.stage]

    t_start = time.time()
    budget_s = args.time_budget_hours * 3600.0
    tasks_cache = {}

    for stage_name in stages_to_run:
        build_fn, run_fn = stages_mod.STAGE_REGISTRY[stage_name]
        all_items = build_fn(zoo)
        shard_items = all_items[args.shard_id::args.num_shards]
        print(f"\n=== STAGE {stage_name}: {len(all_items)} total items, "
              f"{len(shard_items)} in shard {args.shard_id}/{args.num_shards} ===")

        ckpt_name = f"{stage_name}__shard{args.shard_id}of{args.num_shards}.json"
        state_path_in = os.path.join(args.state_dir, ckpt_name)
        state_path_out = os.path.join(args.out_dir, ckpt_name)
        results = load_checkpoint(state_path_in)
        if state_path_in != state_path_out:
            results.update(load_checkpoint(state_path_out))  # prefer out-dir if already partially run there
        print(f"  Loaded {len(results)} cached results")

        n_done_this_run = 0
        n_skipped = 0
        for i, item in enumerate(shard_items):
            key = item_key(item)
            if key in results:
                n_skipped += 1
                continue

            elapsed = time.time() - t_start
            if elapsed > budget_s:
                print(f"  Time budget ({args.time_budget_hours}h) reached, stopping stage {stage_name} "
                      f"at item {i}/{len(shard_items)}")
                save_checkpoint(state_path_out, results)
                print(f"TIME_BUDGET_EXCEEDED stage={stage_name}")
                return

            t0 = time.time()
            try:
                r = run_fn(item, zoo, tasks_cache, device)
                r['_wall_s'] = round(time.time() - t0, 2)
                results[key] = r
                n_done_this_run += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key] = {'_error': str(e)}

            if n_done_this_run % args.save_every == 0:
                save_checkpoint(state_path_out, results)

            if (i + 1) % 20 == 0 or i == len(shard_items) - 1:
                el = time.time() - t_start
                print(f"  [{stage_name}] {i+1}/{len(shard_items)} "
                      f"(done={n_done_this_run} skipped={n_skipped}) elapsed={el/60:.1f}min")

        save_checkpoint(state_path_out, results)
        print(f"  STAGE {stage_name} COMPLETE. {len(results)} total results cached.")

    total_elapsed = time.time() - t_start
    print(f"\nALL REQUESTED STAGES COMPLETE. Total time: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
