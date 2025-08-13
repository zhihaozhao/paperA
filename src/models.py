# Complete src/models.py
# This is a standalone, complete version assuming a typical CSI classification setup.
# It includes definitions for 'enhanced' (as TinyNet example), 'bilstm', and 'cnn'.
# Replace your existing src/models.py with this, or merge it (keep your custom parts).
# Assumes input shape: [batch, T, F] for LSTM (1D seq), [batch, 1, T, F] for CNN (2D).
# Adjust hyperparameters (e.g., hidden_dims) based on your paperA needs.
# Requires: import torch and torch.nn as nn (already included).

import torch
import torch.nn as nn

class TinyNet(nn.Module):  # Example for 'enhanced' - replace with your actual enhanced model if different
    def __init__(self, input_features, num_classes):
        super(TinyNet, self).__init__()
        self.fc1 = nn.Linear(input_features * 128, 256)  # Flatten T=128, F=input_features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten [batch, T, F] -> [batch, T*F]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=8):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional: *2

    def forward(self, x):
        # x: [batch, T, F] (seq_len=T, features=F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        return self.fc(out)

class SimpleCNN(nn.Module):
    def __init__(self, T, F, num_classes=8, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate flattened size: after two pools, height=T//4, width=F//4
        self.fc1 = nn.Linear(64 * (T // 4) * (F // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, T, F] -> Add channel: [batch, 1, T, F]
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def build_model(name, F, num_classes, T=128):  # T added for CNN
    if name == "enhanced":
        return TinyNet(F, num_classes)  # Replace with your actual enhanced if needed

    elif name == "bilstm":
        return BiLSTM(input_dim=F, num_classes=num_classes)

    elif name == "cnn":
        return SimpleCNN(T=T, F=F, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model {name}")

# Optional: Function to count parameters (add to your train_eval.py if needed)
# print(f"Model: {name} with {sum(p.numel() for p in model.parameters())} params")

