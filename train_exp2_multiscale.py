"""
APPROACH 1A: Multi-Scale Temporal Windows
Train CNN-LSTM models on different window sizes (24hr, 48hr, 72hr)
to test if longer windows reveal bipolar cycling patterns.

Hypothesis: Bipolar shows multi-day cycling (48hr+), Unipolar shows sustained flatness
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from data_loader import DepresjonDataLoader
from model import CNNLSTMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


class MultiScaleTrainer:
    """Train separate CNN-LSTM models for each window size."""

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("findings/1A")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.window_sizes = {
            '24hr': 1440,
            '48hr': 2880,
            '72hr': 4320,
        }
        print(f"\nDevice: {self.device}")

    def load_data_for_window(self, window_minutes, window_name):
        """Load and prepare data for a specific window size."""
        loader = DepresjonDataLoader()

        X, y, metadata = loader.create_experiment_dataset(
            experiment=2,
            window_minutes=window_minutes,
            stride_minutes=int(window_minutes // 2)  # 50% overlap
        )

        print(f"\n{window_name} ({window_minutes} min):")
        print(f"  Total windows: {len(X)}")
        print(f"  Bipolar: {(y==0).sum()}, Unipolar: {(y==1).sum()}")

        if len(X) < 100:
            print(f"  [SKIP] Insufficient windows for {window_name}")
            return None

        # Participant-level train/val/test split
        split = loader.create_participant_level_split(X, y, metadata)

        # SMOTE for training data only
        X_train = split['X_train']
        y_train = split['y_train']

        if len(np.unique(y_train)) > 1:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        else:
            X_train_smote, y_train_smote = X_train, y_train

        # Convert to tensors
        X_train = torch.FloatTensor(X_train_smote).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(split['X_val']).unsqueeze(1).to(self.device)
        X_test = torch.FloatTensor(split['X_test']).unsqueeze(1).to(self.device)

        y_train = torch.LongTensor(y_train_smote).to(self.device)
        y_val = torch.LongTensor(split['y_val']).to(self.device)
        y_test = torch.LongTensor(split['y_test']).to(self.device)

        print(f"  After SMOTE: {len(X_train)} training windows")
        print(f"  Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'metadata': split
        }

    def train_model(self, X_train, y_train, X_val, y_val, window_name):
        """Train CNN-LSTM on given window size."""

        model = CNNLSTMClassifier(num_classes=2).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        batch_size = 16
        num_epochs = 50
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n  Training {window_name} model...")

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs} - Train loss: {total_loss:.4f}, Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.results_dir / f'best_model_{window_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test set."""
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_proba = torch.softmax(test_logits, dim=1).cpu().numpy()
            test_pred = np.argmax(test_proba, axis=1)

        y_test_np = y_test.cpu().numpy()

        accuracy = accuracy_score(y_test_np, test_pred)
        cm = confusion_matrix(y_test_np, test_pred)

        try:
            roc_auc = roc_auc_score(y_test_np, test_proba[:, 1])
        except:
            roc_auc = np.nan

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'y_pred': test_pred.tolist(),
            'y_proba': test_proba.tolist(),
        }

    def run_all_windows(self):
        """Train and evaluate models for all window sizes."""

        print("\n" + "="*70)
        print("APPROACH 1A: Multi-Scale Temporal Windows (CNN-LSTM)")
        print("="*70)

        all_results = {}

        for window_name, window_minutes in self.window_sizes.items():
            print(f"\n[{window_name}] Starting...")

            # Load data
            data = self.load_data_for_window(window_minutes, window_name)
            if data is None:
                continue

            # Train
            model = self.train_model(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                window_name
            )

            # Evaluate
            result = self.evaluate_model(model, data['X_test'], data['y_test'])
            all_results[window_name] = result

            print(f"  Accuracy: {result['accuracy']:.4f} | ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  Confusion Matrix:\n{np.array(result['confusion_matrix'])}")

        # Summary and save
        self._print_and_save_summary(all_results)

    def _print_and_save_summary(self, all_results):
        """Print comparison and save results."""

        print("\n" + "="*70)
        print("SUMMARY: Multi-Scale Results Comparison")
        print("="*70)

        results_summary = {}
        best_accuracy = 0
        best_window = None

        for window_name in sorted(all_results.keys()):
            result = all_results[window_name]
            accuracy = result['accuracy']
            roc_auc = result['roc_auc']

            results_summary[window_name] = {
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
                'cm': result['confusion_matrix'],
            }

            print(f"{window_name:8s}: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_window = window_name

        print(f"\nBest window size: {best_window} with {best_accuracy:.4f} accuracy")

        # Test hypothesis: larger windows > smaller windows
        if '24hr' in all_results and '48hr' in all_results:
            acc_24 = all_results['24hr']['accuracy']
            acc_48 = all_results['48hr']['accuracy']
            if acc_48 > acc_24:
                print(f"  [YES] Hypothesis supported: 48hr ({acc_48:.4f}) > 24hr ({acc_24:.4f})")
            else:
                print(f"  [NO] Hypothesis NOT supported: 48hr ({acc_48:.4f}) <= 24hr ({acc_24:.4f})")

        # Save results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '1A - Multi-Scale Windows',
            'window_results': results_summary,
            'best_window': best_window,
            'best_accuracy': float(best_accuracy),
        }

        json_path = self.results_dir / "results_1a.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[SAVED] Results saved to {json_path}")

        return summary


if __name__ == '__main__':
    trainer = MultiScaleTrainer()
    trainer.run_all_windows()
