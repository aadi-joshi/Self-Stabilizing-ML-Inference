# Instructions for Compiling the Paper

1. **Markdown summary**: See `FINAL_PAPER_SUMMARY.md` for a readable, condensed version of the paper with all main results, tables, and section summaries.

2. **Full LaTeX source**: The folder `latex/` contains `paper.tex`, a direct copy of your full LaTeX draft. Copy all figures, .bib files, and any additional resources from your original `stability_constrained_selfimprovement/paper/` folder if needed.

3. **To compile locally**:
   - Install MacTeX (macOS) or TeX Live (Linux/Windows).
   - Run:
     ```
     pdflatex paper.tex
     bibtex paper.aux
     pdflatex paper.tex
     pdflatex paper.tex
     ```
   - Or upload the contents of `latex/` to Overleaf for online compilation.

4. **Figures and Data**:
   - If your paper uses external figures, copy them into the `latex/` folder as well.
   - For Overleaf, upload all referenced images and .bib files.

5. **Results and Data**:
   - All experiment results are in `stability_constrained_selfimprovement/results/20260214_135338/`.

---

If you need a zipped export or further automation, let me know!