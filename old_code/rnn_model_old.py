# rnn_model.py

import os, json, math, errno, time, datetime
import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Bio‑inspired RNN cell with Dale's law (weights sign‑constrained) and autapse mask
# -----------------------------------------------------------------------------

class BioRNN(nn.Module):
    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        n_in, n_rec = hp['n_input'], hp['n_rnn']
        rng = np.random.RandomState(hp['seed'])

        # ---------------- Input weights ----------------
        w_in = rng.randn(n_in, n_rec) / np.sqrt(n_in)
        self.w_in = nn.Parameter(torch.tensor(w_in, dtype=torch.float32), requires_grad=hp['train_w_in'])

        # ---------------- Recurrent weights -------------
        if hp['w_rec_init'] == 'diag':
            w_rec0 = hp['w_rec_gain'] * np.eye(n_rec)
        elif hp['w_rec_init'] == 'randortho':
            temp = rng.randn(n_rec, n_rec)
            w_rec0, _ = np.linalg.qr(temp)
            w_rec0 *= hp['w_rec_gain']
        elif hp['w_rec_init'] == 'randgauss':
            w_rec0 = hp['w_rec_gain'] * rng.randn(n_rec, n_rec) / np.sqrt(n_rec)
        else:
            raise ValueError('Unknown w_rec_init option')

        # --- Dale's law ---------------------------------------------------
        if hp['Dales_law']:
            n_ex = int(n_rec * hp['ei_cell_ratio'])
            n_inh = n_rec - n_ex
            ei_vec = np.concatenate([np.ones(n_ex), -np.ones(n_inh)]).astype(np.float32)
            self.register_buffer('ei_mask', torch.diag(torch.tensor(ei_vec)))
            w_rec0 = np.abs(w_rec0)
            w_rec0[n_ex:, :] *= hp['ei_cell_ratio'] / (1 - hp['ei_cell_ratio'])
        else:
            self.register_buffer('ei_mask', torch.eye(n_rec))

        # Zero autapses (no self‑connections)
        autapse_mask_np = np.ones((n_rec, n_rec), dtype=np.float32)
        np.fill_diagonal(autapse_mask_np, 0.)
        self.register_buffer('autapse_mask', torch.tensor(autapse_mask_np))

        self.w_rec = nn.Parameter(torch.tensor(w_rec0, dtype=torch.float32))

        # ---------------- Bias --------------------------
        self.b_rec = nn.Parameter(torch.zeros(n_rec), requires_grad=hp['train_bias_rec'])

        # ---------------- Non‑linearity -----------------
        act_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
        if hp['activation'] not in act_map:
            raise ValueError('Unsupported activation')
        self.act = act_map[hp['activation']]

        # Constants
        self.alpha = hp['alpha']
        self.sigma = np.sqrt(2 / hp['alpha']) * hp['sigma_rec']

    # ---------------------------------------------------------------------
    def forward(self, x):
        """Run the RNN over the full sequence.
        Args:
            x: (batch, T, n_input)
        Returns:
            h_all: (batch, T, n_rec)
        """
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.hp['n_rnn'], device=x.device)
        h_all = []
        for t in range(T):
            inp = x[:, t] @ self.w_in                      # (batch, n_rec)
            rec = h @ (self.ei_mask @ (self.w_rec * self.autapse_mask))
            noise = torch.randn_like(h) * self.sigma
            h = (1 - self.alpha) * h + self.alpha * self.act(inp + rec + self.b_rec + noise)
            h_all.append(h)
        return torch.stack(h_all, dim=1)                  # (batch, T, n_rec)

# -----------------------------------------------------------------------------
# Complete model = BioRNN → Linear projection
# -----------------------------------------------------------------------------

class RNNModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.rnn = BioRNN(hp)
        self.output = nn.Linear(hp['n_rnn'], hp['n_output'], bias=False)

    def forward(self, x):
        h = self.rnn(x)
        return self.output(h)