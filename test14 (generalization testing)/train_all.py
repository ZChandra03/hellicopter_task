# train_all.py ─ trains GRU / LSTM / RNN on both normal and Bayes targets
# ----------------------------------------------------------------------
import os, json, time, ast
from typing import List, Type

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------
#  Model classes
# ----------------------------------------------------------------------
from rnn_models import GRUModel, LSTMModel, RNNModel      # unchanged

# ----------------------------------------------------------------------
#  Globals
# ----------------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR  = os.path.join(BASE_DIR, "variants")
MAX_VARIANTS = 40

# ----------------------------------------------------------------------
#  Hyper-parameter template
# ----------------------------------------------------------------------
def get_default_hp():
    return dict(
        n_input       = 1,
        n_rnn         = 128,
        batch_size    = 32,
        epochs        = 100,
        learning_rate = 3e-4,
        target_loss   = 2e-3,
    )

# ----------------------------------------------------------------------
#  Dataset that adapts to “normal” vs “Bayes” CSV schemas
# ----------------------------------------------------------------------
class HelicopterDataset(Dataset):
    """
    For *normal* CSVs we expect columns: evidence, trueReport, trueHazard
    For *Bayes*  CSVs we expect columns: evidence, rep_norm,   haz_norm
    Set `is_bayes=True` when loading the latter.
    """
    def __init__(self, csv_path: str, *, is_bayes: bool):
        df = pd.read_csv(csv_path)
        self.x, self.y_rep, self.y_haz = [], [], []

        for _, row in df.iterrows():
            evid = torch.tensor(ast.literal_eval(row["evidence"]),
                                dtype=torch.float32).unsqueeze(-1)
            if is_bayes:        # _bayes variant
                rep = 0.5 * (float(row["rep_norm"]) + 1)     # −1,+1 → 0,1
                haz = float(row["haz_norm"])
            else:               # ordinary variant
                rep = 0.5 * (float(row["trueReport"]) + 1)
                haz = float(row["trueHazard"])

            self.x.append(evid)
            self.y_rep.append(torch.tensor([rep], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz], dtype=torch.float32))

    def __len__(self):             return len(self.x)
    def __getitem__(self, idx):     return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ----------------------------------------------------------------------
#  Train one variant
# ----------------------------------------------------------------------
def train_variant(model: nn.Module, hp: dict, csv_path: str, *,
                  is_bayes: bool) -> float:
    ds  = HelicopterDataset(csv_path, is_bayes=is_bayes)
    dl  = DataLoader(ds, batch_size=hp["batch_size"],
                     shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        model.train()
        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)
            opt.zero_grad()
            logits_rep, logits_haz = model(x)
            loss = (bce(logits_rep[:, -1], y_rep) +
                    0.5 * bce(logits_haz, y_haz))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        if epoch_loss < hp["target_loss"]:
            return epoch_loss   # early stop
    return epoch_loss

# ----------------------------------------------------------------------
#  Train a model class (GRU/LSTM/RNN) across all variants
# ----------------------------------------------------------------------
def train_model_type(model_cls: Type[nn.Module], tag: str, *,
                     is_bayes: bool, variant_suffix: str):
    print(f"\n===== Training {tag} ({'Bayes' if is_bayes else 'normal'}) =====")
    hp    = get_default_hp()
    model = model_cls(hp).to(DEVICE)
    torch.manual_seed(0)

    # bookkeeping
    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    losses, times, best_loss = [], [], float("inf")
    t_start = time.time()

    for k in range(MAX_VARIANTS):
        csv_path = os.path.join(
            VARIANT_DIR, f"trainConfig_var{k}{variant_suffix}.csv")
        if not os.path.isfile(csv_path):
            print(f"Variant {csv_path} not found — stopping.")
            break

        print(f"\n--- Variant {k} ---")
        t0 = time.time()
        v_loss = train_variant(model, hp, csv_path, is_bayes=is_bayes)
        times.append(time.time() - t0)
        losses.append(float(v_loss))

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> new best loss {best_loss:.4e} (checkpoint saved)")

    # persist metadata
    for name, obj in [("loss_history.json", losses),
                      ("times.json",       times),
                      ("hp.json",          hp)]:
        with open(os.path.join(model_dir, name), "w") as f:
            json.dump(obj, f)

    print(f"Finished {tag}: {time.time() - t_start:.1f}s | best {best_loss:.4e}")

# ----------------------------------------------------------------------
#  Main driver: run six trainings
# ----------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_SPECS = [
        (GRUModel,  "gru_trained"),
        (LSTMModel, "lstm_trained"),
        (RNNModel,  "rnn_trained"),
    ]

    for is_bayes, suffix in [(False, ""), (True, "_bayes")]:
        for cls, base_tag in MODEL_SPECS:
            tag = base_tag + ("_bayes" if is_bayes else "")
            train_model_type(cls, tag,
                             is_bayes=is_bayes,
                             variant_suffix=suffix)

    print("\nAll six trainings complete.")
