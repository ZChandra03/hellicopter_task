#!/usr/bin/env python3
"""
hazard_dist_barplot.py  – BARS AT EXACT HAZARD VALUES (0.0 … 1.0)
-----------------------------------------------------------------
Walks through  variants/test/<category>/, grabs the first 20 CSVs in each
category, extracts the true hazard per trial, and plots a grouped bar chart
with bars centred on 0.0, 0.1, …, 1.0.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT = os.path.join(BASE_DIR, "variants", "test")

CATEGORIES = {
    "inf" : ("informative",   "test_informative_*.csv"),
    "unin": ("uninformative", "test_uninformative_*.csv"),
    "mis" : ("misleading",    "test_misleading_*.csv"),
    "uns" : ("unsorted",      "test_unsorted_*.csv"),
}
HAZ_COLS = ("h_true", "hazard", "trueHazard", "hazardRate")

# ------------------------------------------------------------------ #
# Pull hazards from the first 20 CSVs per category
hazards = {}
for key, (folder, pattern) in CATEGORIES.items():
    csvs = sorted(glob.glob(os.path.join(VARIANT_ROOT, folder, pattern)))[:10]
    if not csvs:
        raise RuntimeError(f"No CSVs found for {folder}/{pattern}")

    vals = []
    for path in csvs:
        df = pd.read_csv(path)
        for col in HAZ_COLS:
            if col in df.columns:
                vals.append(df[col].astype(float).values)
                break
        else:
            raise KeyError(f"No recognised hazard column in {path}")
    hazards[key] = np.concatenate(vals)

# ------------------------------------------------------------------ #
# Count (rounded-to-0.1) hazard occurrences
bins      = np.round(np.arange(0, 1.01, 0.1), 1)      # 0.0 … 1.0 inclusive
bin_index = {v: i for i, v in enumerate(bins)}        # map to column index

counts = {key: np.zeros(len(bins), dtype=int) for key in hazards}
for key, arr in hazards.items():
    rounded = np.round(arr, 1)                        # 0.03 → 0.0, 0.27 → 0.3 …
    for h in rounded:
        if h in bin_index:                            # ignore any odd values
            counts[key][bin_index[h]] += 1

# ------------------------------------------------------------------ #
# Plot – bars centred *on* 0.0, 0.1, …, 1.0
fig, ax     = plt.subplots(figsize=(10, 6))
bar_w       = 0.02                                    # thin bars → easy grouping
x_pos       = bins                                    # 0.0 … 1.0
offset_mult = (-1.5, -0.5, 0.5, 1.5)                  # 4 categories → 4 offsets

palette = {"inf": "tab:blue", "unin": "tab:orange",
           "mis": "tab:green", "uns": "tab:red"}

for (key, offs) in zip(palette, offset_mult):
    ax.bar(x_pos + offs*bar_w, counts[key],
           width=bar_w, label=key, color=palette[key])

ax.set_xticks(bins)
ax.set_xlabel("Hazard rate")
ax.set_ylabel("Number of trials\n(first 20 CSVs per category)")
ax.set_title("Hazard-rate distribution across variant categories")
ax.legend(title="Category")
plt.tight_layout()
# Save & show
out_path = os.path.join(BASE_DIR, "hazard_dist.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()
