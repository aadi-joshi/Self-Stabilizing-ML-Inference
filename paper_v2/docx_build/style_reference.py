#!/usr/bin/env python3
"""Restyle pandoc's default reference.docx to look like a standard academic
paper (Times New Roman, justified body text, numbered-looking headings,
1-inch margins), matching the LaTeX article-class look of the PDF."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

SRC = 'custom-reference.docx'
d = Document(SRC)

FONT = 'Times New Roman'


def set_font(style, name=FONT, size=11, bold=False, italic=False, color=None):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    if color:
        style.font.color.rgb = RGBColor(*color)
    # ensure east-asian / complex-script font attrs also set, avoids Word
    # falling back to a different font for some glyphs
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = rpr.makeelement(qn('w:rFonts'), {})
        rpr.append(rFonts)
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    rFonts.set(qn('w:cs'), name)


# ---- page setup: 1 inch margins, Letter-ish (match geometry margin=1in) ----
for section in d.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# ---- Normal (body text) ----
normal = d.styles['Normal']
set_font(normal, size=11, color=(0, 0, 0))
pf = normal.paragraph_format
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.space_after = Pt(8)
pf.space_before = Pt(0)
pf.line_spacing = 1.0

# ---- Title ----
title = d.styles['Title']
set_font(title, size=18, bold=True, color=(0, 0, 0))
title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(12)
title.paragraph_format.space_before = Pt(0)

# ---- Author ----
author = d.styles['Author']
set_font(author, size=12, bold=False, color=(0, 0, 0))
author.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
author.paragraph_format.space_after = Pt(4)
# pandoc's default reference sets keepNext+keepLines on Author, which chains
# it to whatever paragraph follows (Abstract Title); combined with the long
# Abstract body this pushes the whole Title+Author+Abstract block to page 2,
# leaving page 1 almost blank. Disable both so natural flow applies.
author.paragraph_format.keep_with_next = False
author.paragraph_format.keep_together = False

# ---- Date (unused but style it anyway) ----
date_style = d.styles['Date']
set_font(date_style, size=11, color=(0, 0, 0))
date_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---- Abstract Title ----
abst_title = d.styles['Abstract Title']
set_font(abst_title, size=12, bold=True, color=(0, 0, 0))
abst_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
abst_title.paragraph_format.space_before = Pt(18)
abst_title.paragraph_format.space_after = Pt(6)
abst_title.paragraph_format.keep_with_next = False
abst_title.paragraph_format.keep_together = False

# ---- Abstract body ----
abstract = d.styles['Abstract']
set_font(abstract, size=10.5, color=(0, 0, 0))
abstract.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
abstract.paragraph_format.left_indent = Inches(0.35)
abstract.paragraph_format.right_indent = Inches(0.35)
abstract.paragraph_format.space_after = Pt(12)
abstract.paragraph_format.keep_with_next = False
abstract.paragraph_format.keep_together = False

# ---- Headings ----
h1 = d.styles['Heading 1']
set_font(h1, size=14, bold=True, color=(0, 0, 0))
h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
h1.paragraph_format.space_before = Pt(18)
h1.paragraph_format.space_after = Pt(8)
h1.paragraph_format.keep_with_next = True

h2 = d.styles['Heading 2']
set_font(h2, size=12, bold=True, color=(0, 0, 0))
h2.paragraph_format.space_before = Pt(14)
h2.paragraph_format.space_after = Pt(6)
h2.paragraph_format.keep_with_next = True

h3 = d.styles['Heading 3']
set_font(h3, size=11, bold=True, italic=True, color=(0, 0, 0))
h3.paragraph_format.space_before = Pt(12)
h3.paragraph_format.space_after = Pt(4)
h3.paragraph_format.keep_with_next = True

h4 = d.styles['Heading 4']
set_font(h4, size=11, bold=True, color=(0, 0, 0))
h4.paragraph_format.space_before = Pt(10)
h4.paragraph_format.space_after = Pt(4)
h4.paragraph_format.keep_with_next = True

# ---- Captions ----
for cap_name in ['Caption', 'Table Caption', 'Image Caption']:
    cap = d.styles[cap_name]
    set_font(cap, size=10, bold=False, color=(0, 0, 0))
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(10)

# ---- Bibliography ----
bib = d.styles['Bibliography']
set_font(bib, size=10.5, color=(0, 0, 0))
bib.paragraph_format.left_indent = Inches(0.3)
bib.paragraph_format.first_line_indent = Inches(-0.3)
bib.paragraph_format.space_after = Pt(6)
bib.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# ---- Block Text (used for some pandoc constructs) ----
bt = d.styles['Block Text']
set_font(bt, size=10.5, color=(0, 0, 0))

# ---- Table text style ----
tbl = d.styles['Table']
set_font(tbl, size=10, color=(0, 0, 0))

# ---- Compact (used inside table cells / tight lists by pandoc) ----
compact = d.styles['Compact']
set_font(compact, size=10, color=(0, 0, 0))
compact.paragraph_format.space_after = Pt(2)

d.save('academic-reference.docx')
print('wrote academic-reference.docx')

# ---- Patch the theme's major/minor latin fonts (Aptos Display / Aptos) to
# Times New Roman directly. Word's font resolution for styles that carry
# both an explicit w:ascii and a w:asciiTheme reference can prefer the theme
# font in practice, so leaving the theme as Aptos silently overrides the
# explicit Times New Roman set above for Title/Heading-linked styles. ----
import zipfile
import shutil
import re

TMP = 'academic-reference_patched.docx'
shutil.copy('academic-reference.docx', TMP)

with zipfile.ZipFile('academic-reference.docx', 'r') as zin:
    theme_xml = zin.read('word/theme/theme1.xml').decode('utf-8')

theme_xml = theme_xml.replace('typeface="Aptos Display"', 'typeface="Times New Roman"')
theme_xml = theme_xml.replace('typeface="Aptos"', 'typeface="Times New Roman"')

with zipfile.ZipFile('academic-reference.docx', 'r') as zin, \
     zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/theme/theme1.xml':
            data = theme_xml.encode('utf-8')
        zout.writestr(item, data)

shutil.move(TMP, 'academic-reference.docx')
print('patched theme fonts to Times New Roman')
