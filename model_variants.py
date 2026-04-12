"""
Alternative Model Architectures for Approach 2
===============================================

Architecture 1: Bidirectional LSTM
Architecture 2: Attention LSTM
Architecture 3: 1D-RNN-LSTM
Architecture 4: Ensemble (3 CNN-LSTM models with voting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ARCHITECTURE 1: BIDIRECTIONAL LSTM
# ============================================================================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM processes sequence in both directions.

    Rationale:
    - Standard LSTM only processes forward in time
    - BiLSTM sees past AND future context simultaneously
    - Better for symmetric patterns in actigraphy
    - Effective for depression/bipolar signatures that may have bidirectional structure
    """

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.4):
        super().__init__()

        # CNN feature extraction (same as baseline for fair comparison)
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Bidirectional LSTM: processes sequence in both directions
        # hidden_size doubled because bi-directional concatenates forward & backward
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,  # Key difference: bidirectional=True
            dropout=0.0
        )

        # FC layers now take 256 input (128 * 2 for bidirectional)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),  # Changed from 128 to 256 (bidirectional output)
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
        """Forward pass processing sequence bidirectionally."""
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)

        # BiLSTM output: (seq_len, batch, hidden_size*2)
        lstm_out, (h_n, c_n) = self.bilstm(x)

        # Use final output from BiLSTM (concatenation of both directions)
        x = lstm_out[:, -1, :]  # (batch, 256)

        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.output(x)

        return logits


# ============================================================================
# ARCHITECTURE 2: ATTENTION LSTM
# ============================================================================

class AttentionLayer(nn.Module):
    """Bahdanau-style attention mechanism."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights and apply to LSTM output.

        Args:
            lstm_out: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) - weighted sum of LSTM outputs
        """
        # Compute attention weights
        att_weights = torch.softmax(
            self.attention(lstm_out),  # (batch, seq_len, 1)
            dim=1
        )

        # Apply attention to weight time steps
        context = torch.sum(att_weights * lstm_out, dim=1)  # (batch, hidden_size)
        return context


class AttentionLSTMClassifier(nn.Module):
    """
    LSTM with attention mechanism learns to focus on important time steps.

    Rationale:
    - Not all minutes in actigraphy are equally important
    - Attention learns which times (morning restlessness? sleep disruption?) distinguish bipolar/unipolar
    - More interpretable: can visualize which time steps model attends to
    - Especially useful for 24-hour cycles where only certain phases matter
    """

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.4):
        super().__init__()

        # CNN feature extraction
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Attention mechanism
        self.attention = AttentionLayer(hidden_size=128)

        # FC layers
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
        """Forward pass with attention mechanism."""
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Apply attention to focus on important time steps
        x = self.attention(lstm_out)  # (batch, hidden_size)

        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.output(x)

        return logits


# ============================================================================
# ARCHITECTURE 3: 1D-RNN-LSTM
# ============================================================================

class RNNLSTMClassifier(nn.Module):
    """
    1D-RNN followed by LSTM (instead of CNN).

    Rationale:
    - SimpleRNN lighter than CNN, processes sequence directly
    - RNN (basic recurrent layer) captures temporal dynamics at raw level
    - Follows up with LSTM for long-range dependencies
    - Good for understanding if CNN is necessary or sequential RNN sufficient
    - Fewer parameters = less overfitting on small dataset
    - Different inductive bias: RNN good for continuous temporal flow vs CNN good for local patterns
    """

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.4):
        super().__init__()

        # Instead of CNN, use SimpleRNN for initial temporal processing
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.0
        )

        # Second RNN layer for deeper temporal modeling
        self.rnn2 = nn.RNN(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.0
        )

        # LSTM for long-range dependency
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # FC layers
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
        """Forward pass: RNN -> RNN -> LSTM -> FC."""
        # Input: (batch, 1, 1440) -> squeeze to (batch, 1440, 1) for RNN
        x = x.transpose(1, 2)  # (batch, 1440, 1)

        # First RNN layer
        rnn_out, _ = self.rnn(x)  # (batch, 1440, 64)

        # Second RNN layer
        rnn_out2, _ = self.rnn2(rnn_out)  # (batch, 1440, 128)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(rnn_out2)

        # Use final LSTM hidden state
        x = h_n.squeeze(0)  # (batch, 128)

        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.output(x)

        return logits


# ============================================================================
# ARCHITECTURE 4: ENSEMBLE (3 CNN-LSTM models with voting)
# ============================================================================

class EnsembleCNNLSTMClassifier(nn.Module):
    """
    Ensemble of 3 CNN-LSTM models trained with different random seeds.

    Rationale:
    - Neural networks are sensitive to random initialization
    - Ensemble averaging reduces overfitting and variance
    - Voting mechanism more robust than single model
    - Good for small datasets where randomness matters
    - Can average probabilities or use hard voting
    """

    def __init__(self, num_classes: int = 2, num_models: int = 3):
        super().__init__()
        from model import CNNLSTMClassifier

        self.models = nn.ModuleList([
            CNNLSTMClassifier(num_classes=num_classes)
            for _ in range(num_models)
        ])
        self.num_models = num_models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average predictions from all models.

        Args:
            x: Input tensor

        Returns:
            Averaged logits from all models
        """
        logits_list = []
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)

        # Average logits across all models
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return avg_logits

    def set_model_states(self, state_dicts: list):
        """Load pre-trained states for each model."""
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)


if __name__ == "__main__":
    print("Testing all 4 architecture variants...\n")

    x = torch.randn(4, 1, 1440)

    # Test Bidirectional LSTM
    model1 = BiLSTMClassifier(num_classes=2)
    out1 = model1(x)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"BiLSTM: Output {out1.shape}, Params: {params1:,}")

    # Test Attention LSTM
    model2 = AttentionLSTMClassifier(num_classes=2)
    out2 = model2(x)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Attention LSTM: Output {out2.shape}, Params: {params2:,}")

    # Test 1D-RNN-LSTM
    model3 = RNNLSTMClassifier(num_classes=2)
    out3 = model3(x)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"1D-RNN-LSTM: Output {out3.shape}, Params: {params3:,}")

    # Test Ensemble (requires loading existing models)
    print(f"\nNote: Ensemble model requires pre-trained CNN-LSTM models")
    print(f"Will be initialized during training with different random seeds")
