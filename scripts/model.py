"""
1D-CNN-LSTM architecture for actigraphy-based mood disorder classification.
"""

import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    """
    Hybrid 1D-CNN-LSTM for sequence classification on actigraphy data.

    Architecture:
    - CNN Block 1: Conv1D (64 filters, k=7) + BatchNorm + ReLU + MaxPool
    - CNN Block 2: Conv1D (128 filters, k=5) + BatchNorm + ReLU + MaxPool
    - LSTM: 128 hidden units
    - FC layers: 256 -> 64 -> num_classes
    """

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.4):
        super().__init__()

        # CNN Block 1: 1x1440 -> 64x360
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1440 -> 720
        )

        # CNN Block 2: 64x720 -> 128x360
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 720 -> 360
        )

        # LSTM: processes CNN feature sequence
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # no dropout for single layer
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.output = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, 1, 1440) - 24-hour activity sequence

        Returns:
            logits: (batch_size, num_classes)
        """
        # CNN feature extraction
        x = self.cnn_block1(x)  # -> (batch, 64, 720)
        x = self.cnn_block2(x)  # -> (batch, 128, 360)

        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # -> (batch, 360, 128)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: (1, batch, 128)

        # Use final hidden state
        x = h_n.squeeze(0)  # -> (batch, 128)

        # Fully connected
        x = self.fc1(x)  # -> (batch, 256)
        x = self.fc2(x)  # -> (batch, 64)
        logits = self.output(x)  # -> (batch, num_classes)

        return logits


if __name__ == "__main__":
    # Test model
    model = CNNLSTMClassifier(num_classes=2)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass test
    x = torch.randn(4, 1, 1440)
    out = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
