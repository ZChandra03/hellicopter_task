#!/usr/bin/env python3
"""
save_initialized_models.py
===========================
Saves 70 untrained models across all combinations of:
- 7 weight scales (applied to Kaiming uniform initialization)
- 10 seeds (0-9)

Scales: 2x, 5x, 10x, 1/2x, 1/5x, 1/10x, 1x (baseline)
Total: 7 × 10 = 70 models (saved with different scaled initializations)
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
WEIGHT_SCALES = [
    2.0,      # 1. 2x
    5.0,      # 2. 5x
    10.0,     # 3. 10x
    0.5,      # 4. 1/2x
    0.2,      # 5. 1/5x
    0.1,      # 6. 1/10x
    1.0,      # 7. 1x (baseline)
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
def apply_scaled_kaiming_uniform_init(model: nn.Module, scale: float = 1.0) -> None:
    """Apply Kaiming uniform initialization scaled by a factor."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.kaiming_uniform_(param, a=np.sqrt(5))
                # Scale the weights
                with torch.no_grad():
                    param.mul_(scale)
        elif 'bias' in name:
            if param.ndim >= 1:
                fan_in = param.shape[0]
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(param, -bound, bound)
                # Scale the biases
                with torch.no_grad():
                    param.mul_(scale)

# ─────────────────────────── model saving ───────────────────────────
def save_initialized_model(
    model_cls: Type[nn.Module],
    weight_scale: float,
    seed: int,
) -> None:
    """Initialize and save one model with specified weight scale (no training)."""
    
    # Setup experiment name
    scale_name = f"scale{weight_scale:.1f}".replace(".", "p")
    exp_name = f"{scale_name}_seed{seed}"
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
    model_dir = os.path.join(BASE_DIR, "initialized_models", scale_name, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    initial_ckpt = os.path.join(model_dir, "initial.pt")

    # Initialize model
    model = model_cls(hp).to(DEVICE)
    
    # Apply scaled Kaiming uniform initialization
    apply_scaled_kaiming_uniform_init(model, scale=weight_scale)
    
    # Save initialized model
    torch.save(model.state_dict(), initial_ckpt)
    
    # Save hyperparameters with experiment config
    hp_full = hp.copy()
    hp_full.update({
        "init_type": "kaiming_uniform",
        "weight_scale": weight_scale,
        "seed": seed,
        "trained": False,
    })
    json.dump(hp_full, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    
    print(f"  ✓ Saved to {initial_ckpt}")

# ─────────────────────────── entry-point ───────────────────────────
def main() -> None:
    """Save all 70 initialized models: 7 weight scales × 10 seeds."""
    
    total_models = len(WEIGHT_SCALES) * len(SEEDS)
    print(f"\n{'='*80}")
    print(f"Saving {total_models} initialized models (no training)")
    print(f"  Weight scales: {WEIGHT_SCALES}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*80}\n")
    
    model_count = 0
    for scale in WEIGHT_SCALES:
        for seed in SEEDS:
            model_count += 1
            print(f"[{model_count}/{total_models}]", end=" ")
            save_initialized_model(GRUModel, scale, seed)
    
    print(f"\n{'='*80}")
    print(f"All {total_models} initialized models saved!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()