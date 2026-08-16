import re

path = '../ftr_phase_transition_docx.tex'
text = open(path, encoding='utf-8').read()

lines = text.split('\n')
out = []

sec_num = 0
sub_num = 0
in_appendix = False


def letter(n):
    return chr(ord('A') + n - 1)


for line in lines:
    m = re.match(r'^\\appendix\s*$', line)
    if m:
        in_appendix = True
        sec_num = 0
        out.append(line)
        continue

    m = re.match(r'^(\\section\{)(.*)$', line)
    if m:
        sec_num += 1
        sub_num = 0
        label = letter(sec_num) if in_appendix else str(sec_num)
        out.append(f'{m.group(1)}{label} {m.group(2)}')
        continue

    m = re.match(r'^(\\subsection\{)(.*)$', line)
    if m:
        sub_num += 1
        label = f'{letter(sec_num)}.{sub_num}' if in_appendix else f'{sec_num}.{sub_num}'
        out.append(f'{m.group(1)}{label} {m.group(2)}')
        continue

    out.append(line)

result = '\n'.join(out)
open(path, 'w', encoding='utf-8').write(result)
print(f'numbered through section {sec_num} (appendix={in_appendix})')
