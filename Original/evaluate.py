import numpy as np
import torch

from NormativeModel import BayesianObserver

def evaluate_model(model, data, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        pred = torch.argmax(outputs, dim=-1)
    correct = (pred == labels.flatten()).float()
    return correct, correct.mean().item()

def evaluate_model_next_drop(model, data, true_next_drops, num_bins):
    """
    Evaluates the model's ability to predict the next drop (source switch or stay).
    Assumes true_next_drops is in {0, 1}, where 0 → -1 and 1 → +1.
    Returns: (correct, accuracy, predicted_next_flat)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(data)  # [variants * batch, num_bins]
        pred_bins = torch.argmax(outputs, dim=-1)  # [variants * batch]
        pred_hazards = (pred_bins + 0.5) / num_bins  # Convert bin index to hazard rate

        # Reshape back to [variants, batch]
        variants, batch = data.shape[:2]
        pred_hazards = pred_hazards.view(variants, batch)

        # Determine last source from final observation
        last_obs = data[:, :, -1, 0]  # Shape: [variants, batch]
        last_source = torch.where(last_obs < 0, -1, 1)  # -1 or +1

        # Predict next source: switch if hazard > 0.5
        predicted_next = torch.where(pred_hazards > 0.5, -last_source, last_source)

        # Convert true_next_drops from {0, 1} → {-1, 1}
        true_next = true_next_drops.flatten() * 2 - 1
        predicted_next_flat = predicted_next.flatten()

        correct = (predicted_next_flat == true_next).float()
        accuracy = correct.mean().item()

    return correct, accuracy, predicted_next_flat

def evaluate_model_with_tolerance(model, X, y_true, tolerance=1):
    """
    Computes tolerant accuracy: counts a prediction as correct if it is within ±tolerance bins.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)  # [N, num_bins]
        pred_bins = torch.argmax(outputs, dim=-1)

    y_true_flat = y_true.flatten()
    pred_flat = pred_bins.flatten()

    # Tolerant correctness: within ±tolerance
    correct = (torch.abs(pred_flat - y_true_flat) <= tolerance).float()
    accuracy = correct.mean().item()

    return correct, accuracy

def bin_hazard_rates(hazard_array, num_bins):
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(hazard_array, bin_edges) - 1
    return np.clip(bin_indices, 0, num_bins - 1)

def evaluate_bayesian_accuracy(data_subset, hs, mu1=-1, mu2=1, sigma=0.1, mode = 'predict'):
    correct = []
    for ev_tensor, label_tensor in data_subset:
        ev = ev_tensor.squeeze().numpy().tolist()
        true_val = int(label_tensor.item()) * 2 - 1  # Convert from 0/1 → -1/1
        hazard_rate, _, _, pred = BayesianObserver(ev, mu1, mu2, sigma, hs)
        if mode == 'predict':
            correct.append(pred == true_val)
        else: #report
            correct.append(hazard_rate == true_val)
    return np.mean(correct)

def compute_accuracy_vs_hazard(model, X, y, hazard_rates, hs, y_next_drop):
    bin_centers = (hs[:-1] + hs[1:]) / 2
    y_true = y.numpy() if hasattr(y, 'numpy') else y
    hs = np.asarray(hs)
    y_next_true = y_next_drop.numpy() if hasattr(y_next_drop, 'numpy') else y_next_drop
    y_next_true = y_next_true.flatten() * 2 - 1  # Convert from {0, 1} → {-1, +1}
    #y_next_true = y_next_true.flatten() 

    num_bins = len(hs) - 1
    rnn_acc = np.full(num_bins, np.nan)
    bayes_acc = np.full(num_bins, np.nan)
    bin_counts = np.zeros(num_bins, dtype=int)
    rnn_next_acc = np.full(num_bins, np.nan)
    bayes_next_acc = np.full(num_bins, np.nan)

    model.eval()
    with torch.no_grad():
        outputs = model(X)  # Shape: [N, num_bins]
        preds = torch.argmax(outputs, dim=-1).numpy()

    # Assign bin index to each sample
    bin_indices = np.digitize(hazard_rates, hs) - 1

    # Precompute Bayesian predictions
    bayes_hazards = np.empty_like(y_true)
    bayes_next = np.empty_like(y_true)

    idx = 0
    for i in range(X.size(0) * X.size(1)):
        experiment_idx = i // X.size(1)
        sample_idx = i % X.size(1)

        ev = X[experiment_idx, sample_idx, :, 0].numpy().tolist()
        
        L_haz, _, _, pred = BayesianObserver(ev, mu1=-1, mu2=1, sigma=0.1, hs=hs)

        # Get predicted bin: max of posterior at final timestep
        posterior = L_haz[:, -1]
        predicted_bin = np.argmax(posterior)
        
        bayes_hazards[i] = predicted_bin
        bayes_next[i] = pred

    # Evaluate per-bin accuracy
    for i in range(num_bins):
        mask = bin_indices == i
        indices = np.where(mask)[0]
        count = len(indices)
        bin_counts[i] = count

        if count == 0:
            continue

        # +/- 1 bin tolerance
        rnn_in_range = np.abs(preds[indices] - y_true[indices]) <= 1
        bayes_in_range = np.abs(bayes_hazards[indices] - y_true[indices]) <= 1
        rnn_acc[i] = np.mean(rnn_in_range) * 100
        bayes_acc[i] = np.mean(bayes_in_range) * 100

        # Next drop prediction: bin -> hazard rate -> switch/stay
        rnn_next = np.where(preds[indices] > num_bins // 2, 1, -1)
        true_next = y_next_true[indices]
        rnn_next_acc[i] = np.mean(rnn_next == true_next) * 100
        
        bayes_next_bin = bayes_next[indices]  # Select relevant entries
        bayes_next_acc[i] = np.mean(bayes_next_bin == true_next) * 100

    return bin_centers, rnn_acc, bayes_acc, rnn_next_acc, bayes_next_acc, bin_counts