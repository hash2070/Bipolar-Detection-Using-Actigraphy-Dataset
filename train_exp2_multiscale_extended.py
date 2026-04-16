"""
APPROACH 1A EXTENDED: Complete Hyperparameter Grid Search
Test window_size, weight_decay, and dropout for multi-scale CNN-LSTM
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from itertools import product

from data_loader import DepresjonDataLoader
from model import CNNLSTMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE


class MultiScaleHyperparameterSearch:
    """Comprehensive hyperparameter grid search for multi-scale CNN-LSTM."""

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("findings/1A")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define hyperparameter grid
        self.window_sizes = {
            '24hr': 1440,
            '48hr': 2880,
            '72hr': 4320,
        }
        self.weight_decays = [1e-5, 1e-4, 1e-3]  # Test regularization strength
        self.dropouts = [0.2, 0.4, 0.6]          # Test dropout rates

        print(f"\nDevice: {self.device}")
        print(f"Hyperparameter Grid:")
        print(f"  Window sizes: {list(self.window_sizes.keys())}")
        print(f"  Weight decays: {self.weight_decays}")
        print(f"  Dropout rates: {self.dropouts}")
        print(f"  Total configs: {len(self.window_sizes) * len(self.weight_decays) * len(self.dropouts)} = {3 * 3 * 3}")

    def load_data_for_window(self, window_minutes, window_name):
        """Load and prepare data for a specific window size."""
        loader = DepresjonDataLoader()

        X, y, metadata = loader.create_experiment_dataset(
            experiment=2,
            window_minutes=window_minutes,
            stride_minutes=int(window_minutes // 2)
        )

        if len(X) < 100:
            print(f"  [SKIP] {window_name}: Insufficient windows")
            return None

        split = loader.create_participant_level_split(X, y, metadata)

        # SMOTE for training only
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

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }

    def train_model(self, X_train, y_train, X_val, y_val,
                   window_name, weight_decay, dropout_p):
        """Train CNN-LSTM with specific hyperparameters."""

        # Custom model with dropout_p
        model = CNNLSTMClassifier(num_classes=2, dropout_p=dropout_p).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        batch_size = 16
        num_epochs = 50
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
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

        try:
            roc_auc = roc_auc_score(y_test_np, test_proba[:, 1])
        except:
            roc_auc = np.nan

        cm = confusion_matrix(y_test_np, test_pred)

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
        }

    def run_grid_search(self):
        """Run complete hyperparameter grid search."""

        print("\n" + "="*80)
        print("APPROACH 1A EXTENDED: Hyperparameter Grid Search")
        print("="*80)

        all_results = {}
        total_configs = len(self.window_sizes) * len(self.weight_decays) * len(self.dropouts)
        config_count = 0

        # Grid search
        for window_name, window_minutes in self.window_sizes.items():
            window_results = {}

            print(f"\n[Window: {window_name} ({window_minutes} min)]")
            print(f"{'-'*80}")

            # Load data once per window size
            data = self.load_data_for_window(window_minutes, window_name)
            if data is None:
                continue

            for weight_decay, dropout in product(self.weight_decays, self.dropouts):
                config_count += 1
                config_name = f"wd={weight_decay:.0e}_do={dropout:.1f}"

                print(f"\n  Config {config_count}/{total_configs}: {config_name}")

                try:
                    # Train
                    model = self.train_model(
                        data['X_train'], data['y_train'],
                        data['X_val'], data['y_val'],
                        window_name, weight_decay, dropout
                    )

                    # Evaluate
                    result = self.evaluate_model(model, data['X_test'], data['y_test'])
                    window_results[config_name] = result

                    print(f"    Accuracy: {result['accuracy']:.4f} | ROC-AUC: {result['roc_auc']:.4f}")

                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    window_results[config_name] = {'accuracy': 0, 'roc_auc': np.nan}

            all_results[window_name] = window_results

        # Summary and save
        self._print_and_save_summary(all_results)

    def _print_and_save_summary(self, all_results):
        """Print comprehensive summary and save results."""

        print("\n" + "="*80)
        print("SUMMARY: Complete Hyperparameter Grid Search Results")
        print("="*80)

        results_by_window = {}

        # Find best config overall
        best_accuracy_overall = 0
        best_config_overall = None

        for window_name in sorted(all_results.keys()):
            window_data = all_results[window_name]

            print(f"\n{window_name}:")
            print(f"  {'-'*76}")

            best_accuracy_window = 0
            best_config_window = None

            for config_name in sorted(window_data.keys()):
                result = window_data[config_name]
                accuracy = result['accuracy']

                print(f"    {config_name:30s}: Accuracy={accuracy:.4f}")

                if accuracy > best_accuracy_window:
                    best_accuracy_window = accuracy
                    best_config_window = config_name

                if accuracy > best_accuracy_overall:
                    best_accuracy_overall = accuracy
                    best_config_overall = (window_name, config_name)

            print(f"  Best for {window_name}: {best_config_window} ({best_accuracy_window:.4f})")

            results_by_window[window_name] = {
                'best_config': best_config_window,
                'best_accuracy': float(best_accuracy_window),
                'all_results': {k: {
                    'accuracy': float(v['accuracy']),
                    'roc_auc': float(v['roc_auc']) if not np.isnan(v['roc_auc']) else None,
                } for k, v in window_data.items()}
            }

        print(f"\n{'='*80}")
        print(f"OVERALL BEST:")
        if best_config_overall:
            window, config = best_config_overall
            print(f"  Window: {window}")
            print(f"  Config: {config}")
            print(f"  Accuracy: {best_accuracy_overall:.4f}")
        print(f"{'='*80}\n")

        # Save comprehensive results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '1A Extended - Hyperparameter Grid Search',
            'total_configurations_tested': len(self.window_sizes) * len(self.weight_decays) * len(self.dropouts),
            'results_by_window': results_by_window,
            'best_overall': {
                'window': best_config_overall[0] if best_config_overall else None,
                'config': best_config_overall[1] if best_config_overall else None,
                'accuracy': float(best_accuracy_overall),
            } if best_config_overall else None,
        }

        json_path = self.results_dir / "results_1a_extended.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[SAVED] Extended results saved to {json_path}")

        return summary


if __name__ == '__main__':
    searcher = MultiScaleHyperparameterSearch()
    searcher.run_grid_search()
