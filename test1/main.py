import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NormativeModel import BayesianObserver
from rnn import HazardRNN
from data_manager import load_data, convert_subset_to_tensor
from visualization import print_next_drop_stats, plot_total_accuracy, plot_accuracy_vs_hazard, create_bar_plot, plot_accuracy_vs_tolerance, plot_model_behavior_over_time, plot_network_structure, plot_sample_experiments
from evaluate import compute_accuracy_vs_hazard, bin_hazard_rates, evaluate_bayesian_accuracy, evaluate_model, evaluate_model_with_tolerance, evaluate_model_next_drop


# Training setup
params = {'variants': 40}
# Load data
(
    data, easy, medium, hard, pretest,
    hazards_train, hazards_test,
    hazards_easy, hazards_medium, hazards_hard, hazards_pretest,
    y_train_next, y_test_next
) = load_data(params)

print('Data loaded from file.')
mode = "hazard"  # or "next_drop"
eval_mode = 'predict' # report or predict: how the models are evaluated regardless of how they are trained
# Compute sliding average accuracy for the model (window size = 100)
window_size = 50
train_split = "all"  # Options: "all", "easy", "pretest"


if mode == "next_drop":
    num_bins = 2
    output_label_type = "source"  # predict left/right
elif mode == "hazard":
    num_bins = 20
    output_label_type = "hazard"  # predict binned hazard rate
else:
    raise ValueError("Unknown mode")
    
X_test, y_test_next = convert_subset_to_tensor(data[1::2], params['variants'])
X_easy, y_easy_next = convert_subset_to_tensor(easy, params['variants'])
X_med, y_med_next = convert_subset_to_tensor(medium, params['variants'])
X_hard, y_hard_next = convert_subset_to_tensor(hard, params['variants'])
X_pretest, y_pretest_next = convert_subset_to_tensor(pretest, params['variants'])

# Select training data based on `train_split`
if train_split == "all":
    X_train, y_train_next = convert_subset_to_tensor(data[::2], params['variants'])
    train_hazards = hazards_train

elif train_split == "easy":
    X_train, y_train_next = convert_subset_to_tensor(easy, params['variants'])
    train_hazards = hazards_easy

elif train_split == "pretest":
    X_train, y_train_next = convert_subset_to_tensor(pretest, params['variants'])
    train_hazards = hazards_pretest

else:
    raise ValueError(f"Unsupported train_split: {train_split}")

# Create output labels
if mode == "hazard":
    y_train = torch.tensor(bin_hazard_rates(hazards_train, num_bins), dtype=torch.long)
    y_test = torch.tensor(bin_hazard_rates(hazards_test, num_bins), dtype=torch.long)
    y_easy = torch.tensor(bin_hazard_rates(hazards_easy, num_bins), dtype=torch.long)
    y_med = torch.tensor(bin_hazard_rates(hazards_medium, num_bins), dtype=torch.long)
    y_hard = torch.tensor(bin_hazard_rates(hazards_hard, num_bins), dtype=torch.long)
    y_pretest = torch.tensor(bin_hazard_rates(hazards_pretest, num_bins), dtype=torch.long)
else:  # mode == "next_drop"
    y_train = y_train_next
    y_test = y_test_next
    y_easy = torch.tensor([label.item() for _, label in easy], dtype=torch.long)
    y_med = torch.tensor([label.item() for _, label in medium], dtype=torch.long)
    y_hard = torch.tensor([label.item() for _, label in hard], dtype=torch.long)
    y_pretest = torch.tensor([label.item() for _, label in pretest], dtype=torch.long)

print('Data processed for classification!')


print("Running Bayesian Model...")
bayesian_cumulative_accuracy = []
bayesian_correct = []

# Run Bayesian Observer on the test set (assuming y_test is the evidence for simplicity)
#hs = np.arange(0, 1, 0.05)
hs = np.linspace(0, 1, num_bins, endpoint=False) + 0.5 / num_bins  # Centered bins
hazard_bin_edges = np.linspace(0, 1, num_bins + 1)

# Convert true hazard values to bin indices
hazard_bin_indices = np.digitize(hazards_test, bins=hazard_bin_edges) - 1
hazard_bin_indices = np.clip(hazard_bin_indices, 0, num_bins - 1)

bayesian_correct = []

for i in range(X_test.size(0) * X_test.size(1)):
    experiment_idx = i // X_test.size(1)
    sample_idx = i % X_test.size(1)

    ev = X_test[experiment_idx, sample_idx, :, 0].numpy().tolist()
    
    L_haz, _, _, _ = BayesianObserver(ev, mu1=-1, mu2=1, sigma=0.1, hs=hs)

    # Get predicted bin: max of posterior at final timestep
    posterior = L_haz[:, -1]
    predicted_bin = np.argmax(posterior)

    # Compare to true binned label
    true_bin = hazard_bin_indices[i]
    is_correct = (predicted_bin == true_bin)
    bayesian_correct.append(is_correct)

# Run per-difficulty evaluation
if eval_mode == 'predict':
    easy_bayes_acc = evaluate_bayesian_accuracy(easy, hs, mode = eval_mode)
    med_bayes_acc = evaluate_bayesian_accuracy(medium, hs, mode = eval_mode)
    hard_bayes_acc = evaluate_bayesian_accuracy(hard, hs, mode = eval_mode)
    pretest_bayes_acc = evaluate_bayesian_accuracy(pretest, hs, mode = eval_mode)
else: #report
    easy_bayes_acc = evaluate_bayesian_accuracy(hazards_easy, hs, mode = eval_mode)
    med_bayes_acc = evaluate_bayesian_accuracy(hazards_medium, hs, mode = eval_mode)
    hard_bayes_acc = evaluate_bayesian_accuracy(hazards_hard, hs, mode = eval_mode)
    pretest_bayes_acc = evaluate_bayesian_accuracy(hazards_pretest, hs, mode = eval_mode)
        
# Convert to numpy array
bayesian_correct = np.array(bayesian_correct, dtype=np.float32)

bayesian_sliding_accuracy = np.convolve(
    bayesian_correct, np.ones(window_size)/window_size, mode='same'
)

print(f"----- BAYESIAN OBSERVER RESULTS -----")
print(f"Easy Accuracy: {easy_bayes_acc * 100:.2f}%")
print(f"Medium Accuracy: {med_bayes_acc * 100:.2f}%")
print(f"Hard Accuracy: {hard_bayes_acc * 100:.2f}%")
print(f"Pretest Accuracy: {pretest_bayes_acc * 100:.2f}%")

print("Training RNN...")

training_mode = "graded_bins"  # options: "hazard_bins", "graded_bins", "next_drop", "observer"
training_scheme = "instant"  # or "silent"
grading_scheme = [1.0, 0.8, 0.3, 0.1]  # Center bin, +/-1, +/-2, etc...
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01

# Select input and label tensors based on training subset
if train_split == "all":
    X_train_use = X_train
    y_train_use = y_train
    y_train_next_use = y_train_next
elif train_split == "easy":
    X_train_use = X_easy
    y_train_use = y_easy
    y_train_next_use = y_easy_next
elif train_split == "pretest":
    X_train_use = X_pretest
    y_train_use = y_pretest
    y_train_next_use = y_pretest_next
else:
    raise ValueError("Invalid training subset specified.")
    
# Set loss and target labels
if training_mode in ["hazard_bins", "graded_bins"]:
    train_labels = y_train_use
    num_classes = num_bins
elif training_mode == "next_drop":
    train_labels = y_train_next_use
    num_classes = 2
elif training_mode == "observer":
    raise NotImplementedError("Observer mode training is not supported yet.")
else:
    raise ValueError("Unknown training_mode")
    
# Define model, loss, and optimizer
model = HazardRNN(num_bins=num_bins)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 100
# --- TRAIN ---
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    if training_scheme == "instant":
        outputs = model(X_train)  # Shape: [batch, num_classes]

        if training_mode == "graded_bins":
            # Create soft target distribution
            target_probs = torch.zeros_like(outputs)
            for i, true_bin in enumerate(train_labels):
                for offset, weight in enumerate(grading_scheme):
                    for sign in [-1, 1]:
                        if offset == 0 and sign == -1:
                            # Only apply center weight once
                            idx = true_bin
                        else:
                            idx = true_bin + sign * offset
            
                        if 0 <= idx < num_classes:
                            try:
                                target_probs[i, idx] += weight
                            except:
                                exit(0)
            
                # Normalize the row to create a probability distribution
                target_probs[i] = target_probs[i] / target_probs[i].sum()
            # Normalize to make valid distribution
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
            loss = F.kl_div(F.log_softmax(outputs, dim=-1), target_probs, reduction='batchmean')
            
        elif training_mode == "observer":
            outputs = model(X_train)  # Shape: [batch, num_bins]
            target_probs = torch.zeros_like(outputs)
        
            for i in range(X_train.shape[0]):
                ev = X_train[i].squeeze().cpu().numpy()
                L_haz, _, _, _ = BayesianObserver(ev, mu1=-1, mu2=1, sigma=0.1, hs=hs)
                posterior = L_haz[:, -1]
                posterior /= posterior.sum()  # Normalize
                target_probs[i] = torch.tensor(posterior, dtype=torch.float32)
        
            loss = F.kl_div(F.log_softmax(outputs, dim=-1), target_probs.to(outputs.device), reduction='batchmean')
        else:
            # Standard cross-entropy
            loss = criterion(outputs, train_labels)

    elif training_scheme == "silent":
        outputs = model(X_train, return_all=True)  # Shape: [batch, seq_len, num_classes]

        if training_mode != "hazard_bins":
            raise NotImplementedError("Silent mode only implemented for hazard_bins.")

        target_probs = torch.zeros_like(outputs)
        one_hot = F.one_hot(train_labels, num_classes=num_classes).float()
        target_probs[:, -1, :] = one_hot
        loss = F.kl_div(outputs.log(), target_probs, reduction='batchmean')

    else:
        raise ValueError("Unknown training_scheme")

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    out = model(X_train[:100])
    print("Mean prediction:", torch.argmax(out, dim=1).bincount())
    
print("Training complete.")

# Compare predictions with true labels
correct, accuracy = evaluate_model(model, X_test, y_test) 
easy_correct, easy_accuracy = evaluate_model(model, X_easy, y_easy) 
med_correct, med_accuracy  = evaluate_model(model, X_med, y_med) 
hard_correct, hard_accuracy = evaluate_model(model, X_hard, y_hard) 
pretest_correct, pretest_accuracy = evaluate_model(model, X_pretest, y_pretest) 

print(f"----- RNN: EXACT HAZARD RATE ACCURACY -----")
print(f"Total Accuracy: {accuracy * 100:.2f}%")
print(f"Easy Accuracy: {easy_accuracy * 100:.2f}%")
print(f"Medium Accuracy: {med_accuracy * 100:.2f}%")
print(f"Hard Accuracy: {hard_accuracy * 100:.2f}%")
print(f"Pretest Accuracy: {pretest_accuracy * 100:.2f}%")

correct_next, accuracy_next, predicted_next = evaluate_model_next_drop(model, X_test, y_test_next, num_bins)
easy_correct_next, easy_accuracy_next, predicted_easy_next = evaluate_model_next_drop(model, X_easy, y_easy_next, num_bins)
med_correct_next, med_accuracy_next, predicted_med_next = evaluate_model_next_drop(model, X_med, y_med_next, num_bins)
hard_correct_next, hard_accuracy_next, predicted_hard_next = evaluate_model_next_drop(model, X_hard, y_hard_next, num_bins)
pretest_correct_next, pretest_accuracy_next, predicted_pretest_next = evaluate_model_next_drop(model, X_pretest, y_pretest_next, num_bins)

# --- Print ---
print("----- RNN: PREDICT NEXT DROP ACCURACY -----")
print_next_drop_stats("Total", accuracy_next, predicted_next)
print_next_drop_stats("Easy", easy_accuracy_next, predicted_easy_next)
print_next_drop_stats("Medium", med_accuracy_next, predicted_med_next)
print_next_drop_stats("Hard", hard_accuracy_next, predicted_hard_next)
print_next_drop_stats("Pretest", pretest_accuracy_next, predicted_pretest_next)

tolerance = 1
_, accuracy_tol = evaluate_model_with_tolerance(model, X_test, y_test, tolerance) 
_, easy_accuracy_tol = evaluate_model_with_tolerance(model, X_easy, y_easy, tolerance) 
_, med_accuracy_tol  = evaluate_model_with_tolerance(model, X_med, y_med, tolerance) 
_, hard_accuracy_tol = evaluate_model_with_tolerance(model, X_hard, y_hard, tolerance) 
_, pretest_accuracy_tol = evaluate_model_with_tolerance(model, X_pretest, y_pretest, tolerance) 

print(f"----- RNN: HAZARD RATE ACCURACY +/- 1 BIN -----")
print(f"Total Accuracy: {accuracy_tol * 100:.2f}%")
print(f"Easy Accuracy: {easy_accuracy_tol * 100:.2f}%")
print(f"Medium Accuracy: {med_accuracy_tol * 100:.2f}%")
print(f"Hard Accuracy: {hard_accuracy_tol * 100:.2f}%")
print(f"Pretest Accuracy: {pretest_accuracy_tol * 100:.2f}%")

# Pad with zeros on both sides for centered window
model_sliding_accuracy = F.avg_pool1d(correct.view(1, 1, -1), kernel_size=window_size, stride=1, padding=window_size//2)
model_sliding_accuracy = model_sliding_accuracy.squeeze().numpy()


print("Generating Results...")

# Sliding Accuracy Plot
plot_total_accuracy(model_sliding_accuracy, bayesian_sliding_accuracy)

# Sample Prediction Visualization (inferred next drop)
plot_sample_experiments(X_test, y_test_next, model, num_bins)

# Choose a sample index to visualize detailed behavior
sample_idx = 1
flat_index = 0 * X_test.size(1) + sample_idx  # Index into flattened y_train

# Get one trial's input (shape: [20, 1])
evidence = X_test[0, sample_idx]  # [20, 1]
evidence_input = evidence.unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 1]

target_class = y_train[flat_index].item()

# Forward pass with return_all=True
model.eval()
with torch.no_grad():
    outputs = model(evidence_input, return_all=True)  # [1, seq_len, num_bins]
    outputs = outputs.squeeze(0)  # [seq_len, num_bins]

# Create target probs: all 0s except final step (one-hot)
target_probs = torch.full_like(outputs, 0.0)
target_probs[-1] = F.one_hot(torch.tensor(target_class), num_classes=num_bins).float()

# Compute per-timestep loss (KL divergence)
loss_per_timestep = F.kl_div(outputs.log(), target_probs, reduction='none').sum(dim=1).numpy()

# Plot model internals (loss, output, input)
plot_model_behavior_over_time(loss_per_timestep, outputs, evidence)

# Plot structure
# This isn't showing anything useful right now - let's revisit this later
# plot_network_structure(model, num_bins)

# Bar Plot Across Difficulty Levels
labels = ['easy', 'medium', 'hard', 'preTest']
rnn_acc = [easy_accuracy_next * 100, med_accuracy_next * 100, hard_accuracy_next * 100, pretest_accuracy_next * 100]
bayes_acc = [easy_bayes_acc * 100, med_bayes_acc * 100, hard_bayes_acc * 100, pretest_bayes_acc * 100]
create_bar_plot(labels, rnn_acc, bayes_acc)

# Accuracy vs Hazard Rate (all difficulty levels)
with torch.no_grad():
    i=0
    s1 = model(X_test[i:i+1]).squeeze().numpy()
    i=1
    s2 = model(X_test[i:i+1]).squeeze().numpy()
    i=2
    s3 = model(X_test[i:i+1]).squeeze().numpy()
    i=3
    s4 = model(X_test[i:i+1]).squeeze().numpy()
    
# Total
#print("Evaluating performance over all test data...")
#bin_centers, rnn_acc, bayes_acc, rnn_next_acc, bayes_next_acc, bin_counts = compute_accuracy_vs_hazard(model, X_test, y_test, hazards_test, hs, y_test_next)
#for i, (center, acc, count) in enumerate(zip(bin_centers, rnn_acc, bin_counts)):
#    print(f"Bin {i} ({center:.2f}): RNN Accuracy = {acc:.2f}%, Trials = {count}")
#plot_accuracy_vs_hazard(bin_centers, rnn_acc, bayes_acc, bin_counts, title="Total Hazard Bin Accuracy", line=5)
#plot_accuracy_vs_hazard(bin_centers, rnn_next_acc, bayes_next_acc, bin_counts, title="Next Drop Accuracy", line=50)

# Easy
print("Evaluating performance over easy difficulty data...")
bin_centers, rnn_acc_easy, bayes_acc_easy, rnn_next_easy, bayes_next_easy, easy_bin_counts = compute_accuracy_vs_hazard(model, X_easy, y_easy, hazards_easy, hs, y_easy_next)
plot_accuracy_vs_hazard(bin_centers, rnn_acc_easy, bayes_acc_easy, easy_bin_counts, title="Easy Hazard Bin Accuracy", line=5)
plot_accuracy_vs_hazard(bin_centers, rnn_next_easy, bayes_next_easy, easy_bin_counts, title="Easy Next Drop Accuracy", line=50)

# Medium
print("Evaluating performance over medium difficulty data...")
bin_centers, rnn_acc_med, bayes_acc_med, rnn_next_med, bayes_next_med, med_bin_counts = compute_accuracy_vs_hazard(model, X_med, y_med, hazards_medium, hs, y_med_next)
plot_accuracy_vs_hazard(bin_centers, rnn_acc_med, bayes_acc_med, med_bin_counts, title="Medium Hazard Bin Accuracy", line=5)
plot_accuracy_vs_hazard(bin_centers, rnn_next_med, bayes_next_med, med_bin_counts, title="Medium Next Drop Accuracy", line=50)

# Hard
print("Evaluating performance over hard difficulty data...")
bin_centers, rnn_acc_hard, bayes_acc_hard, rnn_next_hard, bayes_next_hard, hard_bin_counts = compute_accuracy_vs_hazard(model, X_hard, y_hard, hazards_hard, hs, y_hard_next)
plot_accuracy_vs_hazard(bin_centers, rnn_acc_hard, bayes_acc_hard, hard_bin_counts, title="Hard Hazard Bin Accuracy", line=5)
plot_accuracy_vs_hazard(bin_centers, rnn_next_hard, bayes_next_hard, hard_bin_counts, title="Hard Next Drop Accuracy", line=50)

plot_accuracy_vs_tolerance(model, X_test, y_test)
    
print("Complete!")
