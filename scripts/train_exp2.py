"""
Training pipeline for Experiment 2: Bipolar vs. Unipolar Depressive Episodes.
Includes SMOTE for handling class imbalance (8 bipolar vs 15 unipolar).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json

from data_loader import DepresjonDataLoader
from model import CNNLSTMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


class Experiment2Trainer:
    """Train and evaluate bipolar vs. unipolar classification with SMOTE."""

    def __init__(self,
                 num_epochs: int = 50,
                 batch_size: int = 16,  # Smaller batch for smaller dataset
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 10,
                 device: str = None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")

        self.model = CNNLSTMClassifier(num_classes=2).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Apply SMOTE to training set at window level."""
        print(f"Applying SMOTE...")
        print(f"  Before: {np.bincount(y)}")

        # Flatten windows to apply SMOTE
        X_flat = X.reshape(X.shape[0], -1)

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

        # Reshape back
        X_resampled = X_resampled_flat.reshape(-1, X.shape[1])

        print(f"  After:  {np.bincount(y_resampled)}")

        return X_resampled, y_resampled

    def _prepare_dataloaders(self, split: dict, apply_smote: bool = True) -> tuple:
        """Create data loaders with SMOTE applied to training set."""
        X_train = split['X_train']
        y_train = split['y_train']
        X_val = split['X_val']
        y_val = split['y_val']
        X_test = split['X_test']
        y_test = split['y_test']

        # Apply SMOTE to training set
        if apply_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val).unsqueeze(1)
        y_val = torch.LongTensor(y_val)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_test = torch.LongTensor(y_test)

        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train.numpy()),
            y=y_train.numpy()
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: {class_weights}")

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().detach().numpy())
            all_labels.append(y_batch.cpu().detach().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        avg_acc = accuracy_score(
            np.concatenate(all_labels),
            np.concatenate(all_preds)
        )

        return avg_loss, avg_acc

    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().detach().numpy())
                all_labels.append(y_batch.cpu().detach().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = accuracy_score(
            np.concatenate(all_labels),
            np.concatenate(all_preds)
        )

        return avg_loss, avg_acc

    def train(self, split: dict) -> None:
        """Train the model with early stopping."""
        train_loader, val_loader, _ = self._prepare_dataloaders(split, apply_smote=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining for {self.num_epochs} epochs with patience={self.patience}...")

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_exp2.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model_exp2.pt'))
        print("Loaded best model")

    def evaluate(self, split: dict) -> tuple:
        """Evaluate on test set and compute metrics."""
        _, _, test_loader = self._prepare_dataloaders(split, apply_smote=False)

        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)

                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)

        # Compute metrics
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_probs[:, 1]),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        return results, y_true, y_pred, y_probs


def run_experiment_2():
    """Run Experiment 2: Bipolar vs. Unipolar Depressive Episodes."""
    print("=" * 60)
    print("EXPERIMENT 2: Bipolar vs. Unipolar Depressive Episodes")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    loader = DepresjonDataLoader("depresjon/data")
    X_exp2, y_exp2, meta_exp2 = loader.create_experiment_dataset(experiment=2)
    split = loader.create_participant_level_split(X_exp2, y_exp2, meta_exp2)

    # Train
    print("\nTraining model...")
    trainer = Experiment2Trainer(num_epochs=50, batch_size=16, patience=10)
    trainer.train(split)

    # Evaluate
    print("\nEvaluating on test set...")
    results, y_true, y_pred, y_probs = trainer.evaluate(split)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f} (primary metric for imbalanced data)")
    print(f"\nConfusion Matrix:\n{np.array(results['confusion_matrix'])}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/exp2_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != 'confusion_matrix'}, f, indent=2)

    np.save("results/exp2_y_true.npy", y_true)
    np.save("results/exp2_y_pred.npy", y_pred)
    np.save("results/exp2_y_probs.npy", y_probs)

    print("\nResults saved to results/")

    return results, y_true, y_pred, y_probs


if __name__ == "__main__":
    run_experiment_2()
