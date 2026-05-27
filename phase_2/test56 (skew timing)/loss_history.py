import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
MODELS_ROOT = BASE_DIR / "models_OTS"

HEAD_ORDER = ["rep", "haz", "both"]
GROUP_ORDER = ["sigma_1", "sigma_2", "sigma_3"]


def extract_run_info(val_loss_path: Path):
    hp_path = val_loss_path.with_name("hp.json")
    if not hp_path.exists():
        return None

    try:
        hp = json.loads(hp_path.read_text())
        val_hist = json.loads(val_loss_path.read_text())
    except Exception as e:
        print(f"Skipping {val_loss_path}: {e}")
        return None

    train_heads = hp.get("train_heads")
    if train_heads not in HEAD_ORDER:
        return None

    parent_names = [p.name for p in val_loss_path.parents]

    group_key = next(
        (name for name in parent_names if re.fullmatch(r"sigma_\d+(?:p\d+)?", name)),
        None,
    )
    seed_name = next(
        (name for name in parent_names if re.fullmatch(r"seed_\d+", name)),
        None,
    )

    if group_key is None or seed_name is None:
        print(f"Skipping {val_loss_path}: could not infer sigma or seed")
        return None

    seed = int(seed_name.split("_")[1])
    val_hist = np.asarray(val_hist, dtype=float)

    return train_heads, group_key, seed, val_hist


def load_val_histories():
    data = defaultdict(lambda: defaultdict(dict))

    for val_loss_path in MODELS_ROOT.rglob("val_loss_history.json"):
        info = extract_run_info(val_loss_path)
        if info is None:
            continue

        train_heads, group_key, seed, val_hist = info
        data[train_heads][group_key][seed] = val_hist

    return data


def main():
    if not MODELS_ROOT.exists():
        raise FileNotFoundError(f"Could not find {MODELS_ROOT}")

    data = load_val_histories()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)

    for i, head in enumerate(HEAD_ORDER):
        for j, group in enumerate(GROUP_ORDER):
            ax = axes[i][j]
            runs = data.get(head, {}).get(group, {})

            if not runs:
                ax.set_title(f"{head} | {group}\n(no runs found)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Val loss")
                ax.grid(True, alpha=0.3)
                continue

            for seed, hist in sorted(runs.items()):
                epochs = np.arange(1, len(hist) + 1)
                ax.plot(epochs, hist, label=f"seed {seed}")

            ax.set_title(f"train_heads={head}, group={group}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = BASE_DIR / "val_loss_histories_3x3.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()