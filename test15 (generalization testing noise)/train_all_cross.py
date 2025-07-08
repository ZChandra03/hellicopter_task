# train_all.py – trains GRU/LSTM/RNN on normal + Bayes targets
#                  for each of the folders variants_1/2/3
# --------------------------------------------------------------------
import os, json, time, ast
from typing import List, Type

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------
#  Model classes (unchanged)
# --------------------------------------------------------------------
from rnn_models import GRUModel, LSTMModel, RNNModel

# --------------------------------------------------------------------
#  Globals
# --------------------------------------------------------------------
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VARIANT_SETS  = ["variants_1", "variants_2", "variants_3"]   # <── new
MAX_VARIANTS  = 10

# --------------------------------------------------------------------
#  Hyper-parameter template
# --------------------------------------------------------------------
def get_default_hp():
    return dict(
        n_input       = 1,
        n_rnn         = 128,
        batch_size    = 32,
        epochs        = 100,
        learning_rate = 3e-4,
        target_loss   = 2e-3,
    )

# --------------------------------------------------------------------
#  Dataset (same as before)
# --------------------------------------------------------------------
class HelicopterDataset(Dataset):
    """
    For *normal* CSVs we expect columns: evidence, trueReport, trueHazard
    For *Bayes*  CSVs we expect columns: evidence, rep_norm,   haz_norm
    """
    def __init__(self, csv_path: str, *, is_bayes: bool):
        df = pd.read_csv(csv_path)
        self.x, self.y_rep, self.y_haz = [], [], []

        for _, row in df.iterrows():
            evid = torch.tensor(ast.literal_eval(row["evidence"]),
                                dtype=torch.float32).unsqueeze(-1)

            if is_bayes:
                rep = 0.5 * (float(row["rep_norm"]) + 1)  # −1/+1 → 0/1
                haz = float(row["haz_norm"])
            else:
                rep = 0.5 * (float(row["trueReport"]) + 1)
                haz = float(row["trueHazard"])

            self.x.append(evid)
            self.y_rep.append(torch.tensor([rep], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz], dtype=torch.float32))

    def __len__(self):         return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.y_rep[i], self.y_haz[i]

# --------------------------------------------------------------------
#  Train one variant CSV
# --------------------------------------------------------------------
def train_variant(model: nn.Module, hp: dict, csv_path: str, *,
                  is_bayes: bool) -> float:
    ds  = HelicopterDataset(csv_path, is_bayes=is_bayes)
    dl  = DataLoader(ds, batch_size=hp["batch_size"],
                     shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce = nn.BCEWithLogitsLoss()

    for _epoch in range(hp["epochs"]):
        running = 0.0
        model.train()
        for x, y_r, y_h in dl:
            x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
            opt.zero_grad()
            log_r, log_h = model(x)
            loss = bce(log_r[:, -1], y_r) + 0.5 * bce(log_h, y_h)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        epoch_loss = running / len(dl)
        if epoch_loss < hp["target_loss"]:
            return epoch_loss
    return epoch_loss

# --------------------------------------------------------------------
#  Train a model class across one variant-set folder
# --------------------------------------------------------------------
def train_model_type(model_cls: Type[nn.Module],
                     tag: str,
                     variant_dir: str,
                     *,
                     is_bayes: bool,
                     variant_suffix: str):

    print(f"\n==== {tag} on {os.path.basename(variant_dir)} "
          f"({'Bayes' if is_bayes else 'normal'}) ====")

    hp    = get_default_hp()
    model = model_cls(hp).to(DEVICE)
    torch.manual_seed(0)

    # save separate checkpoints per tag
    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")

    best_loss, losses, times = float("inf"), [], []
    start = time.time()

    for k in range(MAX_VARIANTS):
        csv_path = os.path.join(
            variant_dir, f"trainConfig_var{k}{variant_suffix}.csv")
        if not os.path.isfile(csv_path):
            print(f"(variants stop at {k})")
            break

        t0 = time.time()
        v_loss = train_variant(model, hp, csv_path, is_bayes=is_bayes)
        losses.append(float(v_loss))
        times.append(time.time() - t0)

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ▸ new best {best_loss:.4e}")

    # metadata
    for fname, data in [("loss_history.json", losses),
                        ("times.json",       times),
                        ("hp.json",          hp)]:
        with open(os.path.join(model_dir, fname), "w") as f:
            json.dump(data, f)

    print(f"Done {tag}: {time.time() - start:.1f}s | best {best_loss:.4e}")

# --------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_SPECS = [
        (GRUModel,  "gru"),
        (LSTMModel, "lstm"),
        (RNNModel,  "rnn"),
    ]

    for variant_dir in (os.path.join(BASE_DIR, v) for v in VARIANT_SETS):
        for is_bayes, suffix in [(False, ""), (True, "_bayes")]:
            for cls, base_tag in MODEL_SPECS:
                tag = f"{base_tag}_{os.path.basename(variant_dir)}" \
                      + ("_bayes" if is_bayes else "")
                train_model_type(cls,
                                 tag,
                                 variant_dir,
                                 is_bayes=is_bayes,
                                 variant_suffix=suffix)

    print("\nAll 18 trainings complete.")
