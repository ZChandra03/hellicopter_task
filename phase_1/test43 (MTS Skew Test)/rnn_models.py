import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Simple GRU encoder that follows the BioRNN interface
# -----------------------------------------------------------------------------
class GRURNN(nn.Module):
    """
    A thin wrapper around nn.GRU that expects the same hp dict used for BioRNN.

    hp keys required
    ----------------
    n_input      : int  – dimensionality of each x_t
    n_rnn        : int  – hidden size
    bidirectional: bool – optional, default False
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        self.n_in   = hp['n_input']
        self.n_rnn  = hp['n_rnn']
        self.bidir  = hp.get('bidirectional', False)

        # 1-layer GRU; keep batch_first=True so (B, T, ⋯) matches your codebase
        self.gru = nn.GRU(
            input_size=self.n_in,
            hidden_size=self.n_rnn,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidir,
        )

        # convenient handle for downstream code
        self.n_out = self.n_rnn * (2 if self.bidir else 1)

    def forward(self, x, h0=None):
        """
        x  : (batch, T, n_input)
        h0 : (1|2, batch, n_rnn) optional initial hidden state
        returns
        -------
        h_all : (batch, T, n_out)
        """
        h_all, _ = self.gru(x, h0)
        return h_all


# -----------------------------------------------------------------------------
# Simple vanilla RNN encoder matching the BioRNN interface
# -----------------------------------------------------------------------------
class RNNRNN(nn.Module):
    """
    A thin wrapper around nn.RNN that expects the same hp dict used for BioRNN.

    hp keys required
    ----------------
    n_input      : int  – dimensionality of each x_t
    n_rnn        : int  – hidden size
    bidirectional: bool – optional, default False
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        self.n_in   = hp['n_input']
        self.n_rnn  = hp['n_rnn']
        self.bidir  = hp.get('bidirectional', False)

        # 1-layer vanilla RNN; keep batch_first=True so (B, T, ⋯)
        self.rnn = nn.RNN(
            input_size=self.n_in,
            hidden_size=self.n_rnn,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidir,
        )

        # convenient handle for downstream code
        self.n_out = self.n_rnn * (2 if self.bidir else 1)

    def forward(self, x, h0=None):
        """
        x  : (batch, T, n_input)
        h0 : (1|2, batch, n_rnn) optional initial hidden state
        returns
        -------
        h_all : (batch, T, n_out)
        """
        h_all, _ = self.rnn(x, h0)
        return h_all


# -----------------------------------------------------------------------------
# Simple LSTM encoder matching the BioRNN interface
# -----------------------------------------------------------------------------
class LSTMRNN(nn.Module):
    """
    A thin wrapper around nn.LSTM that expects the same hp dict used for BioRNN.

    hp keys required
    ----------------
    n_input      : int  – dimensionality of each x_t
    n_rnn        : int  – hidden size
    bidirectional: bool – optional, default False
    """
    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        self.n_in   = hp['n_input']
        self.n_rnn  = hp['n_rnn']
        self.bidir  = hp.get('bidirectional', False)

        # 1-layer LSTM; keep batch_first=True so (B, T, ⋯)
        self.lstm = nn.LSTM(
            input_size=self.n_in,
            hidden_size=self.n_rnn,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidir,
        )

        # convenient handle for downstream code
        self.n_out = self.n_rnn * (2 if self.bidir else 1)

    def forward(self, x, h0=None):
        """
        x  : (batch, T, n_input)
        h0 : tuple((1|2, batch, n_rnn), (1|2, batch, n_rnn)) optional (h0, c0)
        returns
        -------
        h_all : (batch, T, n_out)
        """
        h_all, _ = self.lstm(x, h0)
        return h_all


# -----------------------------------------------------------------------------
# Complete GRU-based model: GRU encoder → two heads
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    """
    Produces *two* outputs:
      1. loc_logits   – (B, T, 1)  site-1 vs site-2 at every step
      2. haz_logits   – (B, 1)     final hazard (> 0.5? stay vs switch)
    """
    def __init__(self, hp):
        super().__init__()
        self.rnn = GRURNN(hp)

        # --- readout heads ----------------------------------------------------
        self.loc_head   = nn.Linear(self.rnn.n_out, 1, bias=False)
        self.haz_head   = nn.Linear(self.rnn.n_out, 1, bias=False)

    def forward(self, x, h0=None):
        h = self.rnn(x, h0)                 # (B, T, n_out)

        loc_logits = self.loc_head(h)       # (B, T, 1)
        haz_logits = self.haz_head(h[:, -1])# (B, 1) – only last step

        return loc_logits, haz_logits


# -----------------------------------------------------------------------------
# Complete vanilla RNN-based model: RNN encoder → two heads
# -----------------------------------------------------------------------------
class RNNModel(nn.Module):
    """
    Produces *two* outputs:
      1. loc_logits   – (B, T, 1)  site-1 vs site-2 at every step
      2. haz_logits   – (B, 1)     final hazard (> 0.5? stay vs switch)
    """
    def __init__(self, hp):
        super().__init__()
        self.rnn = RNNRNN(hp)

        # --- readout heads ----------------------------------------------------
        self.loc_head   = nn.Linear(self.rnn.n_out, 1, bias=False)
        self.haz_head   = nn.Linear(self.rnn.n_out, 1, bias=False)

    def forward(self, x, h0=None):
        h = self.rnn(x, h0)                 # (B, T, n_out)

        loc_logits = self.loc_head(h)       # (B, T, 1)
        haz_logits = self.haz_head(h[:, -1])# (B, 1) – only last step

        return loc_logits, haz_logits


# -----------------------------------------------------------------------------
# Complete LSTM-based model: LSTM encoder → two heads
# -----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    Produces *two* outputs:
      1. loc_logits   – (B, T, 1)  site-1 vs site-2 at every step
      2. haz_logits   – (B, 1)     final hazard (> 0.5? stay vs switch)
    """
    def __init__(self, hp):
        super().__init__()
        self.rnn = LSTMRNN(hp)

        # --- readout heads ----------------------------------------------------
        self.loc_head   = nn.Linear(self.rnn.n_out, 1, bias=False)
        self.haz_head   = nn.Linear(self.rnn.n_out, 1, bias=False)

    def forward(self, x, h0=None):
        h = self.rnn(x, h0)                 # (B, T, n_out)

        loc_logits = self.loc_head(h)       # (B, T, 1)
        haz_logits = self.haz_head(h[:, -1])# (B, 1) – only last step

        return loc_logits, haz_logits
