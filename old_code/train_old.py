import os, json, math, errno, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from old_code.rnn_model_old import RNNModel
# ------------- external dependency -----------------------------
from old_code.task_old import generate_trials
# ---------------------------------------------------------------

# Ensure model outputs always live under this script's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Hyper‑parameter utilities
# -----------------------------------------------------------------------------

def get_default_hp():
    """Return a dict with the same defaults used in the original TF code."""
    return {
        'activation': 'relu',
        'alpha': 0.2,
        'batch_size': 10,
        'bias_rec_init': 0.,
        'comp_step': 20,
        'Dales_law': True,
        'dataset_size': 400,
        'delay': 500,
        'dt': 10,
        'ei_cell_ratio': 0.8,
        'epochs': 4000,
        'learning_rate': 1e-3,
        'loss_type': 'mse',
        'L1_activity': 0.,
        'L2_activity': 1e-6,
        'L1_weight': 0.,
        'L2_weight': 0.,
        'n_input': 1,
        'n_output': 1,
        'n_rnn': 256,
        'optimizer': 'Adam',
        'sigma_rec': 0.05,
        'sigma_x': 5e-3,
        'std_dur': 200,
        'tone_dur': 20,
        'target_loss': 5e-3,
        'train_bias_rec': False,
        'train_w_in': False,
        'validation_split': 0.,
        'w_rec_gain': 0.1,
        'w_rec_init': 'randortho',
        # fields added at runtime
        # 'rule': ..., 'seed': ...
    }

# -----------------------------------------------------------------------------
# Dataset class (handles noise & shuffling each epoch)
# -----------------------------------------------------------------------------

class IntervalDataset(Dataset):
    def __init__(self, hp: dict, rule: str, noise_on: bool = True):
        self.hp = hp
        self.rule = rule
        self.noise_on = noise_on
        self.refresh_trials()

    def refresh_trials(self):
        trial = generate_trials(self.rule, self.hp, 'random', noise_on=self.noise_on)
        self.x = torch.tensor(trial.x, dtype=torch.float32)
        self.y = torch.tensor(trial.y, dtype=torch.float32)
        self.c_mask = torch.tensor(trial.c_mask, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        if self.noise_on:
            x_sample = x_sample + torch.randn_like(x_sample) * self.hp['sigma_x']
        return x_sample, self.y[idx], self.c_mask[idx]

    def on_epoch_end(self):
        # Called manually from the training loop to regenerate a fresh dataset
        self.refresh_trials()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def build_model(hp):
    return RNNModel(hp).to(DEVICE)


def mse_loss_with_mask(pred, target, mask):
    """Temporal MSE respecting the cost mask (same semantics as TF impl)."""
    return ((pred - target) ** 2 * mask).mean()

# -----------------------------------------------------------------------------
# Training routine (single experiment)
# -----------------------------------------------------------------------------

def train(seed_name, rule, seed, checkpoint_suffix='', hp_override=None, reload_directory=None):
    # ----------------------- IO setup ------------------------------------
    model_dir = os.path.join(BASE_DIR, 'models', seed_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, checkpoint_suffix.lstrip('/')) or 'checkpoint.pt'

    # ----------------------- Hyper‑params --------------------------------
    hp = get_default_hp()
    if hp_override:
        hp.update(hp_override)
    hp.update({'rule': rule, 'seed': seed})

    # ----------------------- Data ---------------------------------------
    dataset = IntervalDataset(hp, rule, noise_on=True)
    dataloader = DataLoader(dataset, batch_size=hp['batch_size'], shuffle=True, drop_last=True)

    # ----------------------- Model  -------------------------------------
    torch.manual_seed(seed)
    model = build_model(hp)
    if reload_directory:
        model.load_state_dict(torch.load(os.path.join(reload_directory, checkpoint_suffix), map_location=DEVICE))

    opt = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    # ----------------------- Training loop ------------------------------
    best_loss = float('inf')
    times, loss_hist = [], []
    start_global = time.time()

    for epoch in range(hp['epochs']):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch, c_mask_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            c_mask_batch = c_mask_batch.to(DEVICE)[:, :, 0:1]

            opt.zero_grad()
            # ---- forward pass ----
            y_hat = model(x_batch)
            data_loss = mse_loss_with_mask(y_hat, y_batch, c_mask_batch)

            # ---- activity regularization ----
            # get hidden rates (batch, T, N)
            h = model.rnn(x_batch)
            l1_act = hp['L1_activity'] * h.abs().mean()
            l2_act = hp['L2_activity'] * (h**2).mean()

            # ---- weight regularization on recurrent kernel ----
            w_rec = model.rnn.w_rec
            l1_w = hp['L1_weight'] * w_rec.abs().sum()
            l2_w = hp['L2_weight'] * (w_rec**2).sum()

            # composite loss
            loss = data_loss + l1_act + l2_act + l1_w + l2_w

            # backward + step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # enforce Dale's law if desired
            with torch.no_grad():
                model.rnn.w_rec.clamp_(min=0.0)

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        loss_hist.append(epoch_loss)
        times.append(time.time() - epoch_start)

        # logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f}")

        # checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_path)

        # early stop
        if epoch_loss < hp['target_loss']:
            print(f"Target loss {hp['target_loss']} reached at epoch {epoch}. Stopping early.")
            break

        # regenerate fresh trials each epoch (mimics on_epoch_end in TF)
        dataset.on_epoch_end()

    # ----------------------- Persist metrics ----------------------------
    with open(os.path.join(model_dir, 'loss.json'), 'w') as f:
        json.dump(loss_hist, f)
    with open(os.path.join(model_dir, 'times.json'), 'w') as f:
        json.dump(times, f)
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp, f)

    print(f"Training finished. Best loss = {best_loss:.6f}")

def run():
    numExp = 1
    SeedBase = 100
    rule_list = ['Interval_Discrim']
    checkpoint_suffix = 'checkpoint.pt'

    for exp in range(numExp):
        seed = SeedBase + exp
        for rule in rule_list:
            seed_name = f'easy_trained'
            #seed_name = f'seed_{seed}'
            train(seed_name, rule, seed, checkpoint_suffix, hp_override={'batch_size': 10})

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run()
