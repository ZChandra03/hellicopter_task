
#!/usr/bin/env python3
"""
hazard_dist_continuous.py
-------------------------
For the **continuous‑hazard** pipeline:

• assumes the folder layout created by *TaskConfig_Generator_Continuous.py*::

      └─ variants/
         ├─ beta_0p1/
         │   ├─ trainConfig_0.csv
         │   └─ ...
         ├─ beta_0p5/
         │   └─ ...
         └─ beta_10/
             └─ ...

• scans every ``beta_*`` sub‑folder;
• loads all **train** configs inside;
• plots an empirical hazard *density* histogram (normalised) **vs** the
  analytic Beta(x,x) PDF;
• writes *one PNG per prior* next to its configs.
"""

from __future__ import annotations

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# ───────────────────────── settings ──────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(SCRIPT_DIR, "variants")
FILE_PATTERN = "trainConfig_*.csv"      # only train; adjust if wanted
BINS         = 100                       # histogram bins
DPI          = 150                      # output resolution
# ─────────────────────────────────────────────────────────────

if not os.path.isdir(VARIANTS_DIR):
    raise FileNotFoundError(f"Expect variants/ next to script, found nothing at {VARIANTS_DIR!r}")

# helper ────────────────────────────────────────────────────
def beta_param_from_folder(foldername: str) -> float | None:
    """Extract the scalar *x* from folder name 'beta_<x>'. Return float or None."""
    m = re.fullmatch(r"beta_(\d+p\d+|\d+)", foldername)
    if not m:
        return None
    return float(m.group(1).replace('p', '.'))

def plot_one_folder(sub_path: str, x: float):
    csvs = sorted(glob.glob(os.path.join(sub_path, FILE_PATTERN)))
    if not csvs:
        print(f"[!] No CSVs in {sub_path} — skipped")
        return

    hazards = pd.concat(
        (pd.read_csv(p, usecols=["trueHazard"]) for p in csvs),
        ignore_index=True
    )["trueHazard"].to_numpy()

    # empirical density
    counts, edges = np.histogram(hazards, bins=BINS, range=(0, 1), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # analytic pdf
    xs = np.linspace(0, 1, 1000, endpoint=True)
    pdf = beta_dist.pdf(xs, x, x)

    # plot
    plt.figure(figsize=(7, 4.5))
    plt.bar(centres, counts, width=1/BINS * 0.9,
            align='center', edgecolor='black', alpha=0.7, label='Empirical density')
    plt.plot(xs, pdf, 'k--', lw=2, label=f'Beta({x},{x}) PDF')

    plt.title(f"Hazard distribution – Beta({x},{x}) (n={len(hazards)})")
    plt.xlabel("Hazard h")
    plt.ylabel("Probability density")
    plt.xlim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(sub_path, f"hazard_density_beta_{str(x).replace('.', 'p')}.png")
    plt.savefig(out_png, dpi=DPI)
    plt.close()
    print(f"[✓] Saved {out_png}")

# drive ────────────────────────────────────────────────────
for fname in sorted(os.listdir(VARIANTS_DIR)):
    x_val = beta_param_from_folder(fname)
    if x_val is None:
        continue
    folder_path = os.path.join(VARIANTS_DIR, fname)
    if os.path.isdir(folder_path):
        plot_one_folder(folder_path, x_val)

print("[✓] All prior folders processed.")
