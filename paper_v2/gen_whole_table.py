#!/usr/bin/env python3
import json, os
HERE = os.path.dirname(__file__)
sig = json.load(open(os.path.join(HERE, 'data', 'dense_sweep_summary.json')))
rows = sorted(sig.items(), key=lambda kv: kv[1]['n_params'])
eol = ' \\\\'
lines = []
lines.append('\\begin{table}[h]')
lines.append('\\centering')
lines.append('\\caption{Full 30-architecture crossover table (CIFAR-10, dense $\\eps$ '
             'sweep, 5 seeds per architecture, 10 for CNN\\_W16/CNN\\_W32).}')
lines.append('\\label{tab:fullcrossover}')
lines.append('\\tiny')
lines.append('\\begin{tabular}{llrrrrr}')
lines.append('\\toprule')
lines.append('Architecture & Family & Params & $\\epsstar$ & 95\\% CI & $R^2$ & $k$ \\\\')
lines.append('\\midrule')
for name, r in rows:
    ci = r['boot_ci95']
    tex_name = name.replace('_', '\\_')
    lines.append(f"{tex_name} & {r['family']} & {r['n_params']:,} & {r['eps_star_sigmoid']:.2f} & "
                 f"[{ci[0]:.2f}, {ci[1]:.2f}] & {r['sigmoid_r_sq']:.3f} & {r['sigmoid_k']:.2f}{eol}")
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\end{table}')
with open(os.path.join(HERE, 'full_crossover_table_whole.tex'), 'w', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
print(f"wrote {len(lines)} lines")
