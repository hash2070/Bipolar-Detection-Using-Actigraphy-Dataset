"""
Experiment 2 - Approach 2d: Ensemble (3 CNN-LSTM models with voting)
==================================================================

HYPERPARAMETER JUSTIFICATION:
- num_epochs: 50 (same as baseline)
  WHY: Each individual model trains with baseline hyperparameters.
       Ensemble benefits come from diversity, not longer training.

- batch_size: 16 (same as baseline)
  WHY: Same as baseline - no change needed.

- learning_rate: 1e-3 (same as baseline)
  WHY: Standard configuration for each individual model.

- weight_decay: 1e-4 (same as baseline)
  WHY: Standard regularization.

- patience: 10 (same as baseline)
  WHY: Early stopping at individual model level.

- num_models: 3 (fixed ensemble size)
  WHY: 3 models balance computational cost vs. variance reduction.
       Odd number helps with tie-breaking in voting.

KEY DIFFERENCE:
- Train 3 SEPARATE models with DIFFERENT RANDOM SEEDS
- Each model trains independently on balanced dataset
- During evaluation: AVERAGE predictions (soft voting)
- Soft voting (averaging logits) more stable than hard voting
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


class Experiment2EnsembleTrainer:
    """Train ensemble of 3 CNN-LSTM models with different random seeds."""

    def __init__(self,
                 num_models: int = 3,
                 num_epochs: int = 50,
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 10,
                 device: str = None):
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")
        print(f"Training ensemble of {num_models} CNN-LSTM models")

        # Initialize num_models separate models
        self.models = []
        for i in range(num_models):
            model = CNNLSTMClassifier(num_classes=2).to(self.device)
            self.models.append(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Model {i+1}: {total_params:,} parameters")

        self.histories = [
            {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            for _ in range(num_models)
        ]

    def _downsample_majority_class(self, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
        np.random.seed(random_state)
        bipolar_mask = (y == 0)
        unipolar_mask = (y == 1)
        n_bipolar = bipolar_mask.sum()
        n_unipolar = unipolar_mask.sum()

        bipolar_indices = np.where(bipolar_mask)[0]
        unipolar_indices = np.where(unipolar_mask)[0]

        downsampled_unipolar_indices = np.random.choice(
            unipolar_indices,
            size=n_bipolar,
            replace=False
        )

        balanced_indices = np.concatenate([bipolar_indices, downsampled_unipolar_indices])
        np.random.shuffle(balanced_indices)

        return X[balanced_indices], y[balanced_indices]

    def _prepare_dataloaders(self, split: dict) -> tuple:
        X_train, y_train = self._downsample_majority_class(split['X_train'], split['y_train'])

        X_train = torch.FloatTensor(X_train).unsqueeze(1)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(split['X_val']).unsqueeze(1)
        y_val = torch.LongTensor(split['y_val'])
        X_test = torch.FloatTensor(split['X_test']).unsqueeze(1)
        y_test = torch.LongTensor(split['y_test'])

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train.numpy()),
            y=y_train.numpy()
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False),
            class_weights
        )

    def _train_epoch(self, model: nn.Module, criterion: nn.Module, optimizer, train_loader: DataLoader) -> tuple:
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().detach().numpy())
            all_labels.append(y_batch.cpu().detach().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        avg_acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
        return avg_loss, avg_acc

    def _validate_epoch(self, model: nn.Module, criterion: nn.Module, val_loader: DataLoader) -> tuple:
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().detach().numpy())
                all_labels.append(y_batch.cpu().detach().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
        return avg_loss, avg_acc

    def train(self, split: dict) -> None:
        train_loader, val_loader, _, class_weights = self._prepare_dataloaders(split)

        for model_idx, model in enumerate(self.models):
            print(f"\n{'='*70}")
            print(f"Training Model {model_idx + 1}/{self.num_models}")
            print(f"{'='*70}")

            # Set random seed for this model (different seed for each)
            torch.manual_seed(42 + model_idx)
            np.random.seed(42 + model_idx)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_val_loss = float('inf')
            patience_counter = 0

            print(f"Training for {self.num_epochs} epochs with patience={self.patience}...")

            for epoch in range(self.num_epochs):
                train_loss, train_acc = self._train_epoch(model, criterion, optimizer, train_loader)
                val_loss, val_acc = self._validate_epoch(model, criterion, val_loader)

                self.histories[model_idx]['train_loss'].append(train_loss)
                self.histories[model_idx]['val_loss'].append(val_loss)
                self.histories[model_idx]['train_acc'].append(train_acc)
                self.histories[model_idx]['val_acc'].append(val_acc)

                print(f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'best_model_exp2_ensemble_{model_idx}.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            model.load_state_dict(torch.load(f'best_model_exp2_ensemble_{model_idx}.pt'))

    def evaluate(self, split: dict) -> tuple:
        _, _, test_loader, _ = self._prepare_dataloaders(split)

        all_ensemble_preds = []
        all_ensemble_probs = []
        all_labels = []

        for model in self.models:
            model.eval()
            model_preds = []
            model_probs = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    logits = model(X_batch)
                    probs = torch.softmax(logits, dim=1)

                    model_preds.append(logits.argmax(dim=1).cpu().numpy())
                    model_probs.append(probs.cpu().numpy())

                    if model == self.models[0]:  # Only collect labels once
                        all_labels.append(y_batch.numpy())

            all_ensemble_preds.append(np.concatenate(model_preds))
            all_ensemble_probs.append(np.concatenate(model_probs))

        y_true = np.concatenate(all_labels)

        # Soft voting: average logits across all models
        avg_probs = np.stack(all_ensemble_probs, axis=0).mean(axis=0)
        y_pred = np.argmax(avg_probs, axis=1)

        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, avg_probs[:, 1]) if len(np.unique(y_true)) > 1 else np.nan,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        return results, y_true, y_pred, avg_probs


def run_experiment_2_ensemble():
    """Run Experiment 2 with Ensemble of 3 CNN-LSTM models."""
    print("=" * 70)
    print("EXPERIMENT 2 - APPROACH 2d: Ensemble (3-Model Voting)")
    print("=" * 70)

    loader = DepresjonDataLoader("depresjon/data")
    X_exp2, y_exp2, meta_exp2 = loader.create_experiment_dataset(experiment=2)
    split = loader.create_participant_level_split(X_exp2, y_exp2, meta_exp2)

    trainer = Experiment2EnsembleTrainer(num_models=3, num_epochs=50, batch_size=16, patience=10)
    trainer.train(split)

    results, y_true, y_pred, y_probs = trainer.evaluate(split)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 - APPROACH 2d RESULTS: Ensemble (3 Models)")
    print("=" * 70)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{np.array(results['confusion_matrix'])}")

    Path("results").mkdir(exist_ok=True)
    with open("results/exp2_approach2d_ensemble_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != 'confusion_matrix'}, f, indent=2)

    np.save("results/exp2_approach2d_ensemble_y_true.npy", y_true)
    np.save("results/exp2_approach2d_ensemble_y_pred.npy", y_pred)
    np.save("results/exp2_approach2d_ensemble_y_probs.npy", y_probs)

    print("\nResults saved to results/exp2_approach2d_ensemble_*")
    return results, y_true, y_pred, y_probs


if __name__ == "__main__":
    run_experiment_2_ensemble()
