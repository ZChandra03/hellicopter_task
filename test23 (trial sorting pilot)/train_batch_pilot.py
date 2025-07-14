#!/usr/bin/env python3
"""
train_batch.py — July 2025 (*variant‑shuffle, CSV‑driven, original HP*)
====================================================================

Train **eight RNN models per seed** using *pre‑generated* CSV trial files under
`variants/train/…`.  The script keeps **exactly the same default hyper‑
parameters and early‑stop schedule** as your original `train_batch.py`, but it
replaces on‑the‑fly trial generation with DataFrame loading and shuffles the
order in which each seed sees the *variant* files so that no two seeds have the
same data stream.

### Folder conventions
```
variants/train/
   ├── informative/train_informative_<v>.csv
   ├── uninformative/train_uninformative_<v>.csv
   ├── misleading/train_misleading_<v>.csv
   └── unsorted/train_unsorted_<v>.csv   # full 900‑trial block
```

### Eight model flavours per seed
| Category           | Supervision |
|--------------------|-------------|
| informative‑300    | Ground truth  |
| informative‑300    | Normative     |
| uninformative‑300  | Ground truth  |
| uninformative‑300  | Normative     |
| misleading‑300     | Ground truth  |
| misleading‑300     | Normative     |
| unsorted‑300       | Ground truth  |
| unsorted‑300       | Normative     |

*For the three sorted categories we always train on **all 300** trials in each
variant.  For the unsorted category we deliberately slice the **first 300 rows**
of each 900‑trial CSV so every model sees an equal amount of data.*

Every **five epochs** we evaluate the model on the **entire 900 trials** of the
matching *unsorted* CSV and append that scalar to `baseline_loss_history.json`
so you can compare learning speed across model types.
"""
# ---------------------------------------------------------------------------
# std libs
import os, json, time, copy, random, ast, glob, sys
from typing import List, Tuple, Type

# third‑party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# project modules
from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver
#!/usr/bin/env python3
"""
train_batch.py — July 2025 (*variant‑shuffle, CSV‑driven, original HP*)
====================================================================

Train **eight RNN models per seed** using *pre‑generated* CSV trial files under
`variants/train/…`.  The script keeps **exactly the same default hyper‑
parameters and early‑stop schedule** as your original `train_batch.py`, but it
replaces on‑the‑fly trial generation with DataFrame loading and shuffles the
order in which each seed sees the *variant* files so that no two seeds have the
same data stream.

### Folder conventions
```
variants/train/
   ├── informative/train_informative_<v>.csv
   ├── uninformative/train_uninformative_<v>.csv
   ├── misleading/train_misleading_<v>.csv
   └── unsorted/train_unsorted_<v>.csv   # full 900‑trial block
```

### Eight model flavours per seed
| Category           | Supervision |
|--------------------|-------------|
| informative‑300    | Ground truth  |
| informative‑300    | Normative     |
| uninformative‑300  | Ground truth  |
| uninformative‑300  | Normative     |
| misleading‑300     | Ground truth  |
| misleading‑300     | Normative     |
| unsorted‑300       | Ground truth  |
| unsorted‑300       | Normative     |

*For the three sorted categories we always train on **all 300** trials in each
variant.  For the unsorted category we deliberately slice the **first 300 rows**
of each 900‑trial CSV so every model sees an equal amount of data.*

Every **five epochs** we evaluate the model on the **entire 900 trials** of the
matching *unsorted* CSV and append that scalar to `baseline_loss_history.json`
so you can compare learning speed across model types.
"""
# ---------------------------------------------------------------------------
# std libs
import os, json, time, copy, random, ast, glob, sys
from typing import List, Tuple, Type

# third‑party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# project modules
from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver

# ---------------------------------------------------------------------------
# Constants (unchanged from original)
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT = os.path.join(BASE_DIR, "variants", "train")
MAX_VARIANTS = 40   # adjust if you generated a different number

# ---------------------------------------------------------------------------
# Helper – normative prediction

def normative_predict(evidence: List[float], sigma: float) -> Tuple[int, float]:
    L_haz, _, rep, _ = BayesianObserver(evidence, MU1, MU2, sigma, HS_GRID.copy())
    haz_mean = float(np.dot(HS_GRID, L_haz[:, -1]))
    return int(rep), haz_mean

# ---------------------------------------------------------------------------
# Hyper‑parameters (identical to original train_batch.py)

def get_default_hp():
    return {
        "n_input"       : 1,
        "n_rnn"         : 128,
        "batch_size"    : 25,
        "max_epochs"    : 10,
        "learning_rate" : 3e-4,
        "target_loss"   : 2e-3,
    }

# ---------------------------------------------------------------------------
# Dataset wrapper for CSV rows
class HelicopterDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([0.5 * (row["trueReport"] + 1)], dtype=torch.float32))
            self.y_haz.append(torch.tensor([row["trueHazard"]], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ---------------------------------------------------------------------------
# Utilities to fetch & shuffle variant file lists per seed

def list_variant_paths(category: str) -> List[str]:
    pat = os.path.join(VARIANT_ROOT, category, f"train_{category}_*.csv")
    paths = sorted(glob.glob(pat))[:MAX_VARIANTS]
    if not paths:
        raise FileNotFoundError(f"No CSVs found for pattern {pat}")
    return paths

# ---------------------------------------------------------------------------
# Training core

def train_model(model_cls: Type[nn.Module], tag: str, use_norm: bool,
                category: str, seed: int = 0):
    """Train one model on the specified trial *category* CSVs."""
    hp = get_default_hp()

    # RNG seeding for reproducibility + variant shuffling
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Pre‑load variant paths & randomise order
    variant_paths = list_variant_paths(category)
    variant_order = rng.permutation(len(variant_paths))

    # Pre‑load baseline unsorted CSV (always the first file after shuffle)
    unsorted_paths = list_variant_paths("unsorted")
    baseline_csv   = unsorted_paths[variant_order[0] % len(unsorted_paths)]
    baseline_df    = pd.read_csv(baseline_csv)

    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce   = nn.BCEWithLogitsLoss()

    best, loss_hist, baseline_hist = float("inf"), [], []
    t0 = time.time()

    for epoch in range(hp["max_epochs"]):
        # Pick variant for this epoch
        csv_path = variant_paths[variant_order[epoch % len(variant_paths)]]
        df = pd.read_csv(csv_path)

        # Slice first 300 rows if training on unsorted category
        if category == "unsorted":
            df = df.iloc[:300]

        # Possibly swap targets for normative labels
        if use_norm:
            for idx, row in df.iterrows():
                evid = row["evidence"]
                if not isinstance(evid, list):
                    evid = ast.literal_eval(str(evid))
                rep_norm, haz_norm = normative_predict(evid, float(row["sigma"]))
                df.at[idx, "trueReport"] = rep_norm
                df.at[idx, "trueHazard"]  = haz_norm

        # Training pass
        dl = DataLoader(HelicopterDataset(df), batch_size=hp["batch_size"],
                        shuffle=True, drop_last=True)
        model.train()
        running = 0.0
        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)
            opt.zero_grad()
            logits_rep, logits_haz = model(x)
            loss = bce(logits_rep[:, -1], y_rep) + 0.5 * bce(logits_haz, y_haz)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        epoch_loss = running / len(dl)
        loss_hist.append(epoch_loss)

        # Baseline probe every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                bl_dl = DataLoader(HelicopterDataset(baseline_df), batch_size=hp["batch_size"])
                tot = 0.0
                for x, y_rep, y_haz in bl_dl:
                    x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)
                    logits_rep, logits_haz = model(x)
                    bl_loss = bce(logits_rep[:, -1], y_rep) + 0.5 * bce(logits_haz, y_haz)
                    tot += bl_loss.item()
                baseline_hist.append(tot / len(bl_dl))

        # Progress print every 10 epochs
        if epoch % 10 == 0 or epoch == hp["max_epochs"] - 1:
            print(f"{tag} | ep {epoch:03d} | train {epoch_loss:.4e}")

        # Checkpoints & early stop
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{tag} early stop @ ep {epoch} (best {best:.4e})")
            break

    # Final save & metadata
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist,      open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(baseline_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,             open(os.path.join(model_dir, "hp.json"),             "w"), indent=2)

    print(f"{tag} done in {time.time()-t0:.1f}s | best {best:.4e}")

# ---------------------------------------------------------------------------
# Launch grid — 8 models × N seeds

def main():
    seeds = range(10)   # modify as needed

    # (class, category, use_norm, suffix)
    SPECS = [
        (GRUModel, "informative",  False, "inf_truth"),
        (GRUModel, "informative",  True,  "inf_norm"),
        (GRUModel, "uninformative",False, "unin_truth"),
        (GRUModel, "uninformative",True,  "unin_norm"),
        (GRUModel, "misleading",   False, "mis_truth"),
        (GRUModel, "misleading",   True,  "mis_norm"),
        (GRUModel, "unsorted",     False, "uns_truth"),
        (GRUModel, "unsorted",     True,  "uns_norm"),
    ]

    for seed in seeds:
        for cls, cat, norm_flag, suff in SPECS:
            tag = f"{suff}_s{seed}"
            train_model(cls, tag, norm_flag, category=cat, seed=seed)

    print("All training complete.")

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
# Constants (unchanged from original)
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT = os.path.join(BASE_DIR, "variants", "train")
MAX_VARIANTS = 2   # adjust if you generated a different number

# ---------------------------------------------------------------------------
# Helper – normative prediction

def normative_predict(evidence: List[float], sigma: float) -> Tuple[int, float]:
    L_haz, _, rep, _ = BayesianObserver(evidence, MU1, MU2, sigma, HS_GRID.copy())
    haz_mean = float(np.dot(HS_GRID, L_haz[:, -1]))
    return int(rep), haz_mean

# ---------------------------------------------------------------------------
# Hyper‑parameters (identical to original train_batch.py)

def get_default_hp():
    return {
        "n_input"       : 1,
        "n_rnn"         : 128,
        "batch_size"    : 25,
        "max_epochs"    : 10,
        "learning_rate" : 3e-4,
        "target_loss"   : 2e-3,
    }

# ---------------------------------------------------------------------------
# Dataset wrapper for CSV rows
class HelicopterDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([0.5 * (row["trueReport"] + 1)], dtype=torch.float32))
            self.y_haz.append(torch.tensor([row["trueHazard"]], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ---------------------------------------------------------------------------
# Utilities to fetch & shuffle variant file lists per seed

def list_variant_paths(category: str) -> List[str]:
    pat = os.path.join(VARIANT_ROOT, category, f"train_{category}_*.csv")
    paths = sorted(glob.glob(pat))[:MAX_VARIANTS]
    if not paths:
        raise FileNotFoundError(f"No CSVs found for pattern {pat}")
    return paths

# ---------------------------------------------------------------------------
# Training core

def train_model(model_cls: Type[nn.Module], tag: str, use_norm: bool,
                category: str, seed: int = 0):
    """Train one model on the specified trial *category* CSVs."""
    hp = get_default_hp()

    # RNG seeding for reproducibility + variant shuffling
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Pre‑load variant paths & randomise order
    variant_paths = list_variant_paths(category)
    variant_order = rng.permutation(len(variant_paths))

    # Pre‑load baseline unsorted CSV (always the first file after shuffle)
    unsorted_paths = list_variant_paths("unsorted")
    baseline_csv   = unsorted_paths[variant_order[0] % len(unsorted_paths)]
    baseline_df    = pd.read_csv(baseline_csv)

    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce   = nn.BCEWithLogitsLoss()

    best, loss_hist, baseline_hist = float("inf"), [], []
    t0 = time.time()

    for epoch in range(hp["max_epochs"]):
        # Pick variant for this epoch
        csv_path = variant_paths[variant_order[epoch % len(variant_paths)]]
        df = pd.read_csv(csv_path)

        # Slice first 300 rows if training on unsorted category
        if category == "unsorted":
            df = df.iloc[:300]

        # Possibly swap targets for normative labels
        if use_norm:
            for idx, row in df.iterrows():
                evid = row["evidence"]
                if not isinstance(evid, list):
                    evid = ast.literal_eval(str(evid))
                rep_norm, haz_norm = normative_predict(evid, float(row["sigma"]))
                df.at[idx, "trueReport"] = rep_norm
                df.at[idx, "trueHazard"]  = haz_norm

        # Training pass
        dl = DataLoader(HelicopterDataset(df), batch_size=hp["batch_size"],
                        shuffle=True, drop_last=True)
        model.train()
        running = 0.0
        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)
            opt.zero_grad()
            logits_rep, logits_haz = model(x)
            loss = bce(logits_rep[:, -1], y_rep) + 0.5 * bce(logits_haz, y_haz)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        epoch_loss = running / len(dl)
        loss_hist.append(epoch_loss)

        # Baseline probe every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                bl_dl = DataLoader(HelicopterDataset(baseline_df), batch_size=hp["batch_size"])
                tot = 0.0
                for x, y_rep, y_haz in bl_dl:
                    x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)
                    logits_rep, logits_haz = model(x)
                    bl_loss = bce(logits_rep[:, -1], y_rep) + 0.5 * bce(logits_haz, y_haz)
                    tot += bl_loss.item()
                baseline_hist.append(tot / len(bl_dl))

        # Progress print every 10 epochs
        if epoch % 10 == 0 or epoch == hp["max_epochs"] - 1:
            print(f"{tag} | ep {epoch:03d} | train {epoch_loss:.4e}")

        # Checkpoints & early stop
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{tag} early stop @ ep {epoch} (best {best:.4e})")
            break

    # Final save & metadata
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist,      open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(baseline_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,             open(os.path.join(model_dir, "hp.json"),             "w"), indent=2)

    print(f"{tag} done in {time.time()-t0:.1f}s | best {best:.4e}")

# ---------------------------------------------------------------------------
# Launch grid — 8 models × N seeds

def main():
    seeds = range(2)   # modify as needed

    # (class, category, use_norm, suffix)
    SPECS = [
        (GRUModel, "informative",  False, "inf_truth"),
        (GRUModel, "informative",  True,  "inf_norm"),
        (GRUModel, "uninformative",False, "unin_truth"),
        (GRUModel, "uninformative",True,  "unin_norm"),
        (GRUModel, "misleading",   False, "mis_truth"),
        (GRUModel, "misleading",   True,  "mis_norm"),
        (GRUModel, "unsorted",     False, "uns_truth"),
        (GRUModel, "unsorted",     True,  "uns_norm"),
    ]

    for seed in seeds:
        for cls, cat, norm_flag, suff in SPECS:
            tag = f"{suff}_s{seed}"
            train_model(cls, tag, norm_flag, category=cat, seed=seed)

    print("All training complete.")

if __name__ == "__main__":
    main()
