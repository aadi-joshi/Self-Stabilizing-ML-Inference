import re
from parse_aux_labels import parse_newlabels

labels = parse_newlabels('../ftr_phase_transition.aux')

path = '../ftr_phase_transition_docx.tex'
text = open(path, encoding='utf-8').read()

# replace every \ref{X} with its literal resolved number/letter from the
# aux file. This bypasses pandoc's own cross-reference field generation
# entirely (which produces fragile Word HYPERLINK/REF fields that can show
# raw field-code text like "{HYPERLINK \l "tab:pretrained" \h}" instead of
# their computed value when the field cache is empty or Word doesn't run
# an update-fields pass before export), so every reference in the docx is
# guaranteed to be plain, correctly-resolved text.
count = 0
missing = []


def replace_ref(m):
    global count
    label = m.group(1)
    if label not in labels:
        missing.append(label)
        return m.group(0)
    count += 1
    return labels[label]


text = re.sub(r'\\ref\{([a-zA-Z0-9_:.\-]+)\}', replace_ref, text)

open(path, 'w', encoding='utf-8').write(text)
print(f'replaced {count} refs')
if missing:
    print('WARNING missing labels:', sorted(set(missing)))

remaining = len(re.findall(r'\\ref\{', text))
print(f'remaining unresolved \\ref{{}}: {remaining}')
