"""Parse ftr_phase_transition.aux's \\newlabel{X}{{NUM}{PAGE}{TITLE}{ANCHOR}{}}
entries into a label -> number/letter dict, handling nested braces in TITLE
(citations, math) via proper brace matching rather than a naive regex."""


def match_group(text, start):
    """text[start] must be '{'. Return (inner_content, index_just_after_close)."""
    assert text[start] == '{'
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start + 1:i - 1], i


def parse_newlabels(aux_path):
    text = open(aux_path, encoding='utf-8').read()
    labels = {}
    marker = '\\newlabel{'
    i = 0
    while True:
        start = text.find(marker, i)
        if start == -1:
            break
        j = start + len(marker) - 1  # index of the '{' that opens the label name
        label, k = match_group(text, j)
        # k now points just past the label-name group; next char is the
        # outer '{' wrapping {num}{page}{title}{anchor}{}
        outer_content, after_outer = match_group(text, k)
        # outer_content is "{num}{page}{title}{anchor}{}" as a raw string;
        # extract just the first sub-group {num} from it
        num, _ = match_group(outer_content, 0)
        labels[label] = num
        i = after_outer
    return labels


if __name__ == '__main__':
    labels = parse_newlabels('../ftr_phase_transition.aux')
    print(f'parsed {len(labels)} labels')
    for k, v in sorted(labels.items()):
        print(f'  {k}: {v}')
