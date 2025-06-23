import torch
import torch.nn as nn

# Define the RNN model
class HazardRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=800, num_bins=2, num_layers=1):
        super(HazardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_bins = num_bins

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_bins)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_all=False):
        x = x.squeeze(-1)  # Shape: [variants, experiments, seq_len, trueVal]

        batch_size, num_experiments, seq_len = x.shape  # Unpack dimensions
        x = x.view(batch_size * num_experiments, seq_len, -1)  # Flatten to [total_batches, seq_len, feature_dim]

        h = torch.zeros(batch_size * num_experiments, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.activation(self.fc_input(x_t) + h)
            out_t = self.fc_output(h)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)  # [batch*experiments, seq_len, output_size]
        
        #if return_all:
        #    logits = self.fc_output(outputs)            # [batch*experiments, seq_len, num_bins]
        #    return self.softmax(logits)
        #else:
        #    logits = self.fc_output(outputs[:, -1, :])  # [batch*experiments, num_bins]
        #    return self.softmax(logits)

        if return_all:
            return self.softmax(outputs)  # Return probs at every time step
        else:
            return self.softmax(outputs[:, -1, :])  # Only final output 
