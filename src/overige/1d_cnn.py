import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DClassifier(nn.Module):
    def __init__(self, input_length: int, num_filters: int = 64, kernel_size: int = 3):
        super(CNN1DClassifier, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,  # Single channel: raw packet sizes
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # preserve length
        )

        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_filters)

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()  # for binary classification

    def forward(self, x):
        # x: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # → (batch_size, 1, sequence_length)
        x = self.relu(self.bn1(self.conv1(x)))  # → (batch_size, num_filters, L)
        x = self.relu(self.bn2(self.conv2(x)))  # → (batch_size, num_filters, L)
        x = self.global_pool(x).squeeze(-1)  # → (batch_size, num_filters)
        x = self.fc(x)  # → (batch_size, 1)
        return self.sigmoid(x).squeeze(-1)  # → (batch_size,)
