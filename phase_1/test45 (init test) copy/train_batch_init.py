#!/usr/bin/env python3
"""
save_initialized_models.py
===========================
Saves 80 untrained models across all combinations of:
- 8 initializations
- 10 seeds (0-9)

Total: 8 × 10 = 80 models (saved with different random initializations only)
"""

from __future__ import annotations
import os
import random
import json
from typing import Type

import numpy as np
import torch
import torch.nn as nn

from rnn_models import GRUModel

# ─────────────────────────── configuration ───────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

# Experimental factors
INIT_TYPES = [
    "kaiming_uniform",      # 1. PyTorch default
    "orthogonal",          # 2. Orthogonal (good for RNNs)
    "xavier_uniform",      # 3. Xavier/Glorot uniform
    "xavier_normal",       # 4. Xavier/Glorot normal
    "kaiming_normal",      # 5. Kaiming normal (He init)
    "sparse",              # 6. Sparse initialization
    "uniform",             # 7. Small uniform [-0.1, 0.1]
    "normal"               # 8. Small normal (0, 0.01)
]
SEEDS = list(range(10))

def get_default_hp() -> dict:
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-2,
        "epochs_per_csv": 4,
        "max_csv": 500,
    }

# ─────────────────────────── initialization helpers ───────────────────────────
def apply_kaiming_uniform_init(model: nn.Module) -> None:
    """Apply Kaiming uniform initialization (PyTorch default for linear/GRU layers)."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.kaiming_uniform_(param, a=np.sqrt(5))
        elif 'bias' in name:
            if param.ndim >= 1:
                fan_in = param.shape[0]
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(param, -bound, bound)

def apply_orthogonal_init(model: nn.Module) -> None:
    """Apply orthogonal initialization to all weights."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_xavier_uniform_init(model: nn.Module) -> None:
    """Apply Xavier (Glorot) uniform initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_xavier_normal_init(model: nn.Module) -> None:
    """Apply Xavier (Glorot) normal initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_kaiming_normal_init(model: nn.Module) -> None:
    """Apply Kaiming normal initialization (good for ReLU networks)."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_sparse_init(model: nn.Module, sparsity: float = 0.1) -> None:
    """Apply sparse initialization (most weights are zero)."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.sparse_(param, sparsity=sparsity)
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_uniform_init(model: nn.Module, a: float = -0.1, b: float = 0.1) -> None:
    """Apply uniform initialization in range [a, b]."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, a, b)
        elif 'bias' in name:
            nn.init.uniform_(param, a, b)

def apply_normal_init(model: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Apply normal (Gaussian) initialization with specified mean and std."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean, std)
        elif 'bias' in name:
            nn.init.zeros_(param)

# ─────────────────────────── model saving ───────────────────────────
def save_initialized_model(
    model_cls: Type[nn.Module],
    init_type: str,
    seed: int,
) -> None:
    """Initialize and save one model with specified configuration (no training)."""
    
    # Setup experiment name
    exp_name = f"{init_type}_seed{seed}"
    print(f"Saving initialized model: {exp_name}")
    
    hp = get_default_hp()

    # Set seeds for reproducible initialization
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Directories
    model_dir = os.path.join(BASE_DIR, "initialized_models", init_type, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    initial_ckpt = os.path.join(model_dir, "initial.pt")

    # Initialize model
    model = model_cls(hp).to(DEVICE)
    
    # Apply initialization
    if init_type == "kaiming_uniform":
        apply_kaiming_uniform_init(model)
    elif init_type == "orthogonal":
        apply_orthogonal_init(model)
    elif init_type == "xavier_uniform":
        apply_xavier_uniform_init(model)
    elif init_type == "xavier_normal":
        apply_xavier_normal_init(model)
    elif init_type == "kaiming_normal":
        apply_kaiming_normal_init(model)
    elif init_type == "sparse":
        apply_sparse_init(model)
    elif init_type == "uniform":
        apply_uniform_init(model)
    elif init_type == "normal":
        apply_normal_init(model)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # Save initialized model
    torch.save(model.state_dict(), initial_ckpt)
    
    # Save hyperparameters with experiment config
    hp_full = hp.copy()
    hp_full.update({
        "init_type": init_type,
        "seed": seed,
        "trained": False,
    })
    json.dump(hp_full, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    
    print(f"  ✓ Saved to {initial_ckpt}")

# ─────────────────────────── entry-point ───────────────────────────
def main() -> None:
    """Save all 80 initialized models: 8 inits × 10 seeds."""
    
    total_models = len(INIT_TYPES) * len(SEEDS)
    print(f"\n{'='*80}")
    print(f"Saving {total_models} initialized models (no training)")
    print(f"  Init types: {INIT_TYPES}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*80}\n")
    
    model_count = 0
    for init in INIT_TYPES:
        for seed in SEEDS:
            model_count += 1
            print(f"[{model_count}/{total_models}]", end=" ")
            save_initialized_model(GRUModel, init, seed)
    
    print(f"\n{'='*80}")
    print(f"All {total_models} initialized models saved!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()