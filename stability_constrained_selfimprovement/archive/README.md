# Archived reports

This folder contains long, generated/iterative research writeups that embed images under paths like `results/neurips_*/plots/...`.

Those plot files are **generated outputs** (and are intentionally **not committed** in the public repo), so leaving these reports at the top level would render a lot of broken images on GitHub.

## How to regenerate plots locally

From `stability_constrained_selfimprovement/`:

```bash
pip install -r requirements.txt

# Example: regenerate a full set of results + plots
python run_complete.py
```

After running, `results/neurips_*/plots/` should exist locally and these archived markdown files will render their images correctly.
