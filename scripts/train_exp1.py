"""
Training pipeline for Experiment 1: Healthy vs. Depressed classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
import time

from data_loader import DepresjonDataLoader
from model import CNNLSTMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


class Experiment1Trainer:
    """Train and evaluate healthy vs. depressed classification."""

    def __init__(self,
                 num_epochs: int = 50,
                 batch_size: int = 32,
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

    def _prepare_dataloaders(self, split: Dict, compute_weights: bool = True) -> Tuple:
        """Create data loaders with optional class weighting."""
        # Convert to tensors
        X_train = torch.FloatTensor(split['X_train']).unsqueeze(1)  # Add channel dim
        y_train = torch.LongTensor(split['y_train'])
        X_val = torch.FloatTensor(split['X_val']).unsqueeze(1)
        y_val = torch.LongTensor(split['y_val'])
        X_test = torch.FloatTensor(split['X_test']).unsqueeze(1)
        y_test = torch.LongTensor(split['y_test'])

        # Compute class weights for imbalanced training set
        if compute_weights:
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

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
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

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
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

    def train(self, split: Dict) -> None:
        """Train the model with early stopping."""
        train_loader, val_loader, _ = self._prepare_dataloaders(split, compute_weights=True)

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
                torch.save(self.model.state_dict(), 'best_model_exp1.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model_exp1.pt'))
        print("Loaded best model")

    def evaluate(self, split: Dict) -> Dict:
        """Evaluate on test set and compute metrics."""
        _, _, test_loader = self._prepare_dataloaders(split, compute_weights=False)

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


def run_experiment_1():
    """Run Experiment 1: Healthy vs. Depressed."""
    print("=" * 60)
    print("EXPERIMENT 1: Healthy vs. Depressed Classification")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    loader = DepresjonDataLoader("depresjon/data")
    X_exp1, y_exp1, meta_exp1 = loader.create_experiment_dataset(experiment=1)
    split = loader.create_participant_level_split(X_exp1, y_exp1, meta_exp1)

    # Train
    print("\nTraining model...")
    trainer = Experiment1Trainer(num_epochs=50, batch_size=32, patience=10)
    trainer.train(split)

    # Evaluate
    print("\nEvaluating on test set...")
    results, y_true, y_pred, y_probs = trainer.evaluate(split)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{np.array(results['confusion_matrix'])}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/exp1_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != 'confusion_matrix'}, f, indent=2)

    np.save("results/exp1_y_true.npy", y_true)
    np.save("results/exp1_y_pred.npy", y_pred)
    np.save("results/exp1_y_probs.npy", y_probs)

    print("\nResults saved to results/")

    return results, y_true, y_pred, y_probs


if __name__ == "__main__":
    run_experiment_1()
