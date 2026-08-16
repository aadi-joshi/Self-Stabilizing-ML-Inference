import re
text = open('../ftr_phase_transition.tex', encoding='utf-8').read()
text2 = re.sub(r'(\\includegraphics\[[^\]]*\]\{figures/[a-zA-Z0-9_]+)\.pdf\}', r'\1.png}', text)
open('../ftr_phase_transition_docx.tex', 'w', encoding='utf-8').write(text2)
print('done')
