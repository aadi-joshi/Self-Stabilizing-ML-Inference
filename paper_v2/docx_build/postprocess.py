#!/usr/bin/env python3
"""Post-process the pandoc-generated docx: remove the manual line break in
the title (LaTeX's \\\\ renders correctly at LaTeX's font metrics but forces
an awkward 3-line wrap at Word's Times New Roman metrics) and let Word wrap
the title naturally instead."""
from docx import Document

d = Document('ftr_paper.docx')

title_para = None
for p in d.paragraphs:
    if p.style.name == 'Title':
        title_para = p
        break

if title_para is not None:
    # collapse the explicit <w:br/> line break into a plain space so Word
    # wraps the full title naturally at its own font metrics
    for run in title_para.runs:
        br_elems = run._element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br')
        for br in br_elems:
            br.getparent().remove(br)
    # merge the two text runs with a space between them
    texts = [r.text for r in title_para.runs if r.text]
    if len(texts) >= 2:
        title_para.runs[0].text = ' '.join(texts)
        for r in title_para.runs[1:]:
            r.text = ''
    print('title line break removed:', title_para.text)
else:
    print('WARNING: no Title-styled paragraph found')

# clean up document metadata: pandoc dumps the whole author block (incl.
# affiliations/emails) into core_properties.author; replace with just names
d.core_properties.author = 'Kavya Bhand, Aadi Joshi, Vijay Rathod'
d.core_properties.subject = 'Continual learning; Lagrangian-dual constrained optimization; functional trust regions'
d.core_properties.comments = ''

d.save('ftr_paper.docx')
print('saved')
