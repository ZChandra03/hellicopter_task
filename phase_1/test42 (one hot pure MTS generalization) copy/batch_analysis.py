import os
import glob
import pandas as pd
from pathlib import Path

# ------------------- user knobs -------------------
GROUP_KEY = "hz_0p3_0p7"   # the group_folder to filter
INCLUDE_NORMATIVE = False   # set True to keep seed == -1 rows
PREFERRED_CSV = "figures_MTS/binned_acc_uniform_per_seed_wide.csv"
# --------------------------------------------------

def locate_per_seed_csv(preferred_path: str) -> str:
    """
    Try preferred path; if missing, search for a reasonable match.
    Search patterns: *per_seed_wide*.csv inside likely roots.
    Choose the most recently modified candidate.
    """
    if os.path.exists(preferred_path):
        print(f"[info] Using CSV: {preferred_path}")
        return preferred_path

    # likely roots: cwd, script dir, its parent, and any figures_* folders
    here = Path(__file__).resolve().parent
    roots = {
        Path.cwd(),
        here,
        here.parent,
        Path.cwd() / "figures_MTS",
        Path.cwd() / "figures",
        here / "figures_MTS",
        here / "figures",
    }

    patterns = ["*per_seed_wide*.csv", "binned_acc_uniform_per_seed_wide.csv"]
    candidates = []
    for root in roots:
        for pat in patterns:
            candidates.extend(glob.glob(str(root / "**" / pat), recursive=True))

    if not candidates:
        tried = [str(Path.cwd() / PREFERRED_CSV)]
        raise FileNotFoundError(
            "Could not find a per-seed wide CSV.\n"
            f"Tried preferred path:\n  - {tried[0]}\n"
            "Also searched for patterns '*per_seed_wide*.csv' under common roots.\n"
            "Make sure you've generated the per-seed file (e.g., "
            "'binned_acc_uniform_per_seed_wide.csv')."
        )

    # pick newest by mtime
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    picked = candidates[0]
    print(f"[info] Found CSV via search: {picked}")
    return picked

def choose_group(df: pd.DataFrame, group_key: str) -> str:
    """Use exact match if present; else allow unique substring match."""
    groups = df["group_folder"].astype(str).unique().tolist()
    if group_key in groups:
        return group_key
    subs = [g for g in groups if group_key in g]
    if len(subs) == 1:
        print(f"[warn] Exact '{group_key}' not found; using closest: '{subs[0]}'")
        return subs[0]
    msg = ["Available groups:"]
    msg += [f"  - {g}" for g in sorted(groups)]
    raise ValueError(f"group_folder '{group_key}' not found.\n" + "\n".join(msg))

def per_seed_accuracy_for_group(csv_path: str, group_key: str, include_normative: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"group_folder", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    group = choose_group(df, group_key)
    sub = df[df["group_folder"].astype(str) == group].copy()

    # Build/report overall from columns if needed
    report_bins = [c for c in sub.columns if c.startswith("report_b")]
    hazard_bins = [c for c in sub.columns if c.startswith("hazard_b")]

    if "report_overall" in sub.columns:
        sub["report_acc"] = pd.to_numeric(sub["report_overall"], errors="coerce")
    else:
        if not report_bins:
            raise ValueError("No report_bXX columns to compute report accuracy.")
        sub["report_acc"] = sub[report_bins].mean(axis=1, skipna=True)

    if "hazard_overall" in sub.columns:
        sub["hazard_acc"] = pd.to_numeric(sub["hazard_overall"], errors="coerce")
    else:
        if not hazard_bins:
            raise ValueError("No hazard_bXX columns to compute hazard accuracy.")
        sub["hazard_acc"] = sub[hazard_bins].mean(axis=1, skipna=True)

    # Drop normative if requested (often seed == -1)
    if not include_normative:
        sub = sub[pd.to_numeric(sub["seed"], errors="coerce").fillna(-1) >= 0]

    # Clean types & sort
    sub["seed"] = pd.to_numeric(sub["seed"], errors="coerce").astype("Int64")
    out = sub[["seed", "report_acc", "hazard_acc"]].sort_values("seed").reset_index(drop=True)
    return out

def main():
    csv_path = locate_per_seed_csv(PREFERRED_CSV)
    out_df = per_seed_accuracy_for_group(csv_path, GROUP_KEY, INCLUDE_NORMATIVE)

    print("\nPer-seed accuracies (mean across bins):")
    print(out_df.to_string(index=False))

    # Save next to source CSV
    out_dir = str(Path(csv_path).resolve().parent)
    out_csv = os.path.join(out_dir, f"per_seed_accuracy_{GROUP_KEY}.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

if __name__ == "__main__":
    main()
