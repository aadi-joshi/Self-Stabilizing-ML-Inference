import re

HERE_PARENT = '..'

text = open(f'{HERE_PARENT}/ftr_phase_transition_docx.tex', encoding='utf-8').read()

# inline the two \input{} table files so everything lives in one file
for fname in ['full_crossover_table_whole.tex', 'curvature_table_whole.tex']:
    content = open(f'{HERE_PARENT}/{fname}', encoding='utf-8').read()
    text = text.replace(f'\\input{{{fname}}}', content)

# walk the document, tracking begin{table}/begin{figure} context, and prefix
# each \caption{...} with "Table N: " / "Figure N: " using brace matching to
# find the caption's true extent (captions contain nested braces from
# \citep, \ref, math, etc., so a naive regex would stop at the first "}")
out = []
i = 0
n = len(text)
table_n = 0
fig_n = 0
cur_kind = None  # 'table' or 'figure', set on begin{...}, cleared on end{...}

while i < n:
    if text.startswith('\\begin{table}', i):
        cur_kind = 'table'
        out.append('\\begin{table}')
        i += len('\\begin{table}')
        continue
    if text.startswith('\\begin{figure}', i):
        cur_kind = 'figure'
        out.append('\\begin{figure}')
        i += len('\\begin{figure}')
        continue
    if text.startswith('\\end{table}', i):
        cur_kind = None
        out.append('\\end{table}')
        i += len('\\end{table}')
        continue
    if text.startswith('\\end{figure}', i):
        cur_kind = None
        out.append('\\end{figure}')
        i += len('\\end{figure}')
        continue
    if text.startswith('\\caption{', i) and cur_kind is not None:
        # find matching closing brace
        start = i + len('\\caption{')
        depth = 1
        j = start
        while j < n and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        inner = text[start:j-1]
        if cur_kind == 'table':
            table_n += 1
            prefix = f'Table {table_n}: '
        else:
            fig_n += 1
            prefix = f'Figure {fig_n}: '
        out.append('\\caption{' + prefix + inner + '}')
        i = j
        continue
    out.append(text[i])
    i += 1

result = ''.join(out)
open(f'{HERE_PARENT}/ftr_phase_transition_docx.tex', 'w', encoding='utf-8').write(result)
print(f'tables numbered: {table_n}, figures numbered: {fig_n}')
