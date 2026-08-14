#!/usr/bin/env python3
"""Merge per-shard checkpoint JSONs into one file per stage."""
import argparse
import glob
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state-dir', default='./state')
    ap.add_argument('--out-dir', default='./state_merged')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pattern = re.compile(r'^(?P<stage>.+)__shard(?P<sid>\d+)of(?P<n>\d+)\.json$')
    by_stage = {}
    for path in glob.glob(os.path.join(args.state_dir, '*__shard*of*.json')):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        stage = m.group('stage')
        by_stage.setdefault(stage, []).append(path)

    for stage, paths in by_stage.items():
        merged = {}
        for p in paths:
            with open(p) as f:
                merged.update(json.load(f))
        out_path = os.path.join(args.out_dir, f'{stage}.json')
        with open(out_path, 'w') as f:
            json.dump(merged, f)
        print(f"{stage}: merged {len(paths)} shard file(s) -> {len(merged)} results -> {out_path}")


if __name__ == '__main__':
    main()
