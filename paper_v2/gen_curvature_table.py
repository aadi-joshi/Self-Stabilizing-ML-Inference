#!/usr/bin/env python3
import json, os
import numpy as np
HERE = os.path.dirname(__file__)
curv_raw = json.load(open(os.path.join(HERE, 'data', 'curvature_full_zoo.json')))
sig = json.load(open(os.path.join(HERE, 'data', 'dense_sweep_summary.json')))

by_arch = {}
for k, v in curv_raw.items():
    if '_error' in v:
        continue
    key = json.loads(k)
    by_arch.setdefault(key['arch'], []).append(v)

rows = []
for arch, entries in by_arch.items():
    ht = np.mean([e['hessian_trace'] for e in entries])
    ft = np.mean([e['fisher_trace'] for e in entries])
    sn = np.mean([e['spectral_norm'] for e in entries])
    deff = np.mean([e['d_eff'] for e in entries])
    n_params = sig[arch]['n_params'] if arch in sig else entries[0]['n_params']
    rows.append((arch, n_params, ht, ft, sn, deff))

rows.sort(key=lambda r: r[1])
eol = ' \\\\'
lines = []
lines.append('\\begin{table}[h]')
lines.append('\\centering')
lines.append('\\caption{Curvature statistics for all 30 architectures (mean over 5 seeds, '
             'measured after Task 1 training, matching the protocol of Section~4.1).}')
lines.append('\\label{tab:curvature}')
lines.append('\\tiny')
lines.append('\\begin{tabular}{lrrrrr}')
lines.append('\\toprule')
lines.append('Architecture & Params & $\\mathrm{tr}(H)$ & $\\mathrm{tr}(F)$ & Spectral norm & $d_{\\mathrm{eff}}$ \\\\')
lines.append('\\midrule')
for arch, n_params, ht, ft, sn, deff in rows:
    tex_name = arch.replace('_', '\\_')
    lines.append(f"{tex_name} & {n_params:,} & {ht:.1f} & {ft:.3f} & {sn:.1f} & {deff:.2f}{eol}")
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')

with open(os.path.join(HERE, 'curvature_table_whole.tex'), 'w', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
print(f"wrote {len(rows)} data rows")
