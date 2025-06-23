import torch
import pandas as pd
import numpy as np
import random

def load_data(params, shuffle=True):
    all_data = []
    easy_data, medium_data, hard_data, pretest_data = [], [], [], []
    hazards_train, hazards_test = [], []
    hazards_easy, hazards_medium, hazards_hard, hazards_pretest = [], [], [], []

    true_next_train, true_next_test = [], []

    for k in range(params['variants']):
        train_data = pd.read_csv(f"variants/trainConfig_var{k}.csv")
        test_data = pd.read_csv(f"variants/testConfig_var{k}.csv")

        # Filter prediction trials only
        train_data = train_data[train_data.iloc[:, 4] == 'predict']
        test_data = test_data[test_data.iloc[:, 4] == 'predict']

        # --- TRAIN ---
        evidence_train = train_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_train = train_data['trueVal'].values
        hazard_train = train_data['trueHazard'].values
        hazards_train.extend(hazard_train.tolist())

        true_next_train.extend((true_val_train == 1).astype(int).tolist())

        all_data.append((
            torch.tensor(np.stack(evidence_train), dtype=torch.float32).unsqueeze(-1),
            torch.tensor((true_val_train == 1).astype(int), dtype=torch.long)
        ))

        # --- TEST ---
        evidence_test = test_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_test = test_data['trueVal'].values
        hazard_test = test_data['trueHazard'].values
        hazards_test.extend(hazard_test.tolist())

        true_next_test.extend((true_val_test == 1).astype(int).tolist())

        all_data.append((
            torch.tensor(np.stack(evidence_test), dtype=torch.float32).unsqueeze(-1),
            torch.tensor((true_val_test == 1).astype(int), dtype=torch.long)
        ))

        # Difficulty-specific breakdown
        difficulty = test_data.iloc[:, 2].values  # Column 3: difficulty
        for ev, val, diff, hz in zip(evidence_test.tolist(), true_val_test, difficulty, hazard_test):
            sample = (
                torch.tensor(ev, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(int(val == 1), dtype=torch.long)
            )
            if diff == 'easy':
                easy_data.append(sample)
                hazards_easy.append(hz)
            elif diff == 'medium':
                medium_data.append(sample)
                hazards_medium.append(hz)
            elif diff == 'hard':
                hard_data.append(sample)
                hazards_hard.append(hz)
            elif diff == 'preTest':
                pretest_data.append(sample)
                hazards_pretest.append(hz)

    # Shuffle each difficulty-specific set if needed
    if shuffle:
        for dataset, hazard_list in [
            (easy_data, hazards_easy),
            (medium_data, hazards_medium),
            (hard_data, hazards_hard),
            (pretest_data, hazards_pretest),
        ]:
            paired = list(zip(dataset, hazard_list))
            random.shuffle(paired)
            dataset[:], hazard_list[:] = zip(*paired)

    return (
        all_data,
        easy_data, medium_data, hard_data, pretest_data,
        np.array(hazards_train), np.array(hazards_test),
        np.array(hazards_easy), np.array(hazards_medium), np.array(hazards_hard), np.array(hazards_pretest),
        torch.tensor(true_next_train, dtype=torch.long),
        torch.tensor(true_next_test, dtype=torch.long)
    )


def convert_subset_to_tensor(subset, variants):
    """
    Converts a list of (evidence_tensor, label_tensor) pairs into X and y tensors
    shaped like X_test and y_test in your workflow.
    """
    X_list, y_list = zip(*subset)
    X_tensor = torch.stack(X_list).unsqueeze(-1)  # Shape: [N, 20, 1, 1]
    y_tensor = torch.stack(y_list)                # Shape: [N]
    
    X_tensor = X_tensor.view(variants, -1, 20, 1)
    
    return X_tensor, y_tensor