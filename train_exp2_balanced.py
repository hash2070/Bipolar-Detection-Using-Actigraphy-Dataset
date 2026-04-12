"""
Experiment 2 - Approach 1: Balanced Class Distribution via Downsampling
=========================================================================
Instead of using SMOTE to upsample bipolar, we downsample unipolar to match
bipolar count. This forces the model to learn the bipolar signal without
relying on synthetic data generation.

Hypothesis: Model collapse in original Exp 2 was due to overwhelming majority
class. With balanced natural data, the model should learn to distinguish.
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


class Experiment2BalancedTrainer:
    """
    Train bipolar vs. unipolar with balanced classes via downsampling.
    No SMOTE - pure class balancing through selective sampling.
    """

    def __init__(self,
                 num_epochs: int = 100,  # More epochs for smaller dataset
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 15,  # More patience for harder task
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

    def _downsample_majority_class(self, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
        """
        Downsample majority class (unipolar, label=1) to match minority class (bipolar, label=0).

        Args:
            X: Feature array
            y: Label array (0 = bipolar, 1 = unipolar)
            random_state: Random seed

        Returns:
            X_balanced, y_balanced: Downsampled data
        """
        np.random.seed(random_state)

        bipolar_mask = (y == 0)
        unipolar_mask = (y == 1)

        n_bipolar = bipolar_mask.sum()
        n_unipolar = unipolar_mask.sum()

        print(f"Before downsampling:")
        print(f"  Bipolar (minority): {n_bipolar} samples")
        print(f"  Unipolar (majority): {n_unipolar} samples")
        print(f"  Imbalance ratio: {n_unipolar / n_bipolar:.1f}x")

        # Get indices for each class
        bipolar_indices = np.where(bipolar_mask)[0]
        unipolar_indices = np.where(unipolar_mask)[0]

        # Downsample unipolar to match bipolar
        downsampled_unipolar_indices = np.random.choice(
            unipolar_indices,
            size=n_bipolar,
            replace=False
        )

        # Combine indices
        balanced_indices = np.concatenate([bipolar_indices, downsampled_unipolar_indices])
        np.random.shuffle(balanced_indices)

        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]

        print(f"\nAfter downsampling:")
        print(f"  Bipolar: {(y_balanced == 0).sum()} samples")
        print(f"  Unipolar: {(y_balanced == 1).sum()} samples")
        print(f"  Imbalance ratio: 1.0x (perfectly balanced)")

        return X_balanced, y_balanced

    def _prepare_dataloaders(self, split: dict) -> tuple:
        """Create data loaders with downsampled training set."""
        X_train = split['X_train']
        y_train = split['y_train']
        X_val = split['X_val']
        y_val = split['y_val']
        X_test = split['X_test']
        y_test = split['y_test']

        # Downsample training set
        print("\nApplying downsampling to training set...")
        X_train, y_train = self._downsample_majority_class(X_train, y_train)

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val).unsqueeze(1)
        y_val = torch.LongTensor(y_val)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_test = torch.LongTensor(y_test)

        # Compute class weights (should be 1.0, 1.0 for balanced data)
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

        print(f"Train set: {len(train_loader.dataset)} windows")
        print(f"Val set:   {len(val_loader.dataset)} windows")
        print(f"Test set:  {len(test_loader.dataset)} windows")

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
        train_loader, val_loader, _ = self._prepare_dataloaders(split)

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
                torch.save(self.model.state_dict(), 'best_model_exp2_balanced.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model_exp2_balanced.pt'))
        print("Loaded best model")

    def evaluate(self, split: dict) -> tuple:
        """Evaluate on test set and compute metrics."""
        _, _, test_loader = self._prepare_dataloaders(split)

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
            'roc_auc': roc_auc_score(y_true, y_probs[:, 1]) if len(np.unique(y_true)) > 1 else np.nan,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        return results, y_true, y_pred, y_probs


def run_experiment_2_balanced():
    """Run Experiment 2 with Approach 1: Downsampled balanced classes."""
    print("=" * 70)
    print("EXPERIMENT 2 - APPROACH 1: Balanced Classes via Downsampling")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    loader = DepresjonDataLoader("depresjon/data")
    X_exp2, y_exp2, meta_exp2 = loader.create_experiment_dataset(experiment=2)
    split = loader.create_participant_level_split(X_exp2, y_exp2, meta_exp2)

    # Train
    print("\nTraining model...")
    trainer = Experiment2BalancedTrainer(num_epochs=100, batch_size=16, patience=15)
    trainer.train(split)

    # Evaluate
    print("\nEvaluating on test set...")
    results, y_true, y_pred, y_probs = trainer.evaluate(split)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 - APPROACH 1 RESULTS: Balanced Class Distribution")
    print("=" * 70)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix (0=Bipolar, 1=Unipolar):")
    print(np.array(results['confusion_matrix']))

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/exp2_approach1_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != 'confusion_matrix'}, f, indent=2)

    np.save("results/exp2_approach1_y_true.npy", y_true)
    np.save("results/exp2_approach1_y_pred.npy", y_pred)
    np.save("results/exp2_approach1_y_probs.npy", y_probs)

    print("\nResults saved to results/exp2_approach1_*")

    return results, y_true, y_pred, y_probs


if __name__ == "__main__":
    results, y_true, y_pred, y_probs = run_experiment_2_balanced()
