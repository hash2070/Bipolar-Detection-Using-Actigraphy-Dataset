"""
APPROACH 1C: Participant-Level Aggregation
Classify bipolar vs unipolar using participant-level variability metrics.
Uses Leave-One-Out cross-validation on 23 mood-disordered participants.

Expected result: 65-70% LOOCV accuracy by capturing bipolar's multi-day variability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import json
from datetime import datetime

class ParticipantVariabilityClassifier:
    def __init__(self, data_path: str = "depresjon/data"):
        """Initialize with path to data directory."""
        from data_loader import DepresjonDataLoader
        self.loader = DepresjonDataLoader(data_path)
        self.results_dir = Path("findings/1C")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def compute_participant_features(self) -> pd.DataFrame:
        """
        Compute variability metrics for each condition participant (bipolar/unipolar).
        Returns DataFrame with one row per participant.

        KEY INSIGHT: Bipolar should have HIGH variability across days (mood cycling),
                    Unipolar should have LOW variability (sustained depression)
        """
        participant_ids = []
        labels = []
        features_list = []

        # Get unique condition participants only (bipolar or unipolar)
        unique_pids = self.loader.metadata[
            self.loader.metadata['is_condition']
        ]['participant_id'].unique()

        print(f"\nProcessing {len(unique_pids)} condition participants...")

        for pid in unique_pids:
            # Get metadata
            meta = self.loader.metadata[
                self.loader.metadata['participant_id'] == pid
            ].iloc[0]

            # Label: 0=Bipolar (afftype 1 or 1.5), 1=Unipolar (afftype 2)
            label = int(meta['afftype'] == 2)
            afftype_name = "Unipolar" if meta['afftype'] == 2 else "Bipolar"

            # Load activity data
            activity_data = self.loader._load_activity(pid)
            activity = activity_data['activity'].values.astype(float)

            # Normalize per participant
            activity = (activity - activity.mean()) / (activity.std() + 1e-8)

            # Compute daily statistics (1440 minutes = 24 hours per day)
            daily_stats = []
            num_days = len(activity) // 1440

            for day in range(num_days):
                day_start = day * 1440
                day_end = day_start + 1440
                if day_end <= len(activity):
                    day_activity = activity[day_start:day_end]
                    daily_stats.append({
                        'day_mean': day_activity.mean(),
                        'day_std': day_activity.std(),
                        'day_max': day_activity.max(),
                        'day_min': day_activity.min(),
                    })

            if len(daily_stats) < 2:
                print(f"  [SKIP] Skipping {pid} ({afftype_name}): insufficient data ({num_days} days)")
                continue

            daily_means = np.array([d['day_mean'] for d in daily_stats])
            daily_stds = np.array([d['day_std'] for d in daily_stats])
            daily_maxs = np.array([d['day_max'] for d in daily_stats])

            # Feature engineering
            mean_activity = np.mean(daily_means)
            variability = np.std(daily_means)  # KEY: Bipolar should have high variability
            activity_range = np.max(daily_means) - np.min(daily_means)
            coef_var = variability / (np.abs(mean_activity) + 1e-8)
            mean_daily_std = np.mean(daily_stds)

            features = {
                'participant_id': pid,
                'afftype': afftype_name,
                'label': label,
                'num_days': len(daily_stats),
                'mean_activity': mean_activity,
                'variability_across_days': variability,
                'range': activity_range,
                'coefficient_of_variation': coef_var,
                'mean_daily_std': mean_daily_std,
                'max_activity': np.max(daily_maxs),
                'min_activity': np.min(daily_means),
            }

            participant_ids.append(pid)
            labels.append(label)
            features_list.append(features)

            print(f"  [OK] {pid:15s} ({afftype_name:8s}): variability={variability:7.3f}, range={activity_range:7.3f}")

        return pd.DataFrame(features_list)

    def train_and_evaluate_loocv(self, features_df: pd.DataFrame,
                                 c_values: list = [0.1, 1.0, 10.0],
                                 class_weights: list = [None, 'balanced']):
        """
        Train logistic regression using Leave-One-Out CV.
        Test different C (regularization) and class_weight values.
        """

        feature_cols = ['mean_activity', 'variability_across_days', 'range',
                       'coefficient_of_variation', 'mean_daily_std']
        X = features_df[feature_cols].values
        y = features_df['label'].values

        results_by_config = {}
        best_accuracy = 0
        best_config = None

        print(f"\n{'='*70}")
        print(f"APPROACH 1C: Logistic Regression LOOCV")
        print(f"{'='*70}")
        print(f"Participants: {len(y)} (Bipolar: {(y==0).sum()}, Unipolar: {(y==1).sum()})")
        print(f"Features: {feature_cols}")
        print(f"\nTesting hyperparameter combinations:")
        print(f"  C (regularization): {c_values}")
        print(f"  class_weight: {class_weights}")
        print(f"\n{'-'*70}\n")

        # Test different hyperparameter combinations
        for c_val in c_values:
            for cw_val in class_weights:
                config_name = f"C={c_val}, class_weight={cw_val}"

                # Leave-One-Out Cross-Validation
                loo = LeaveOneOut()
                y_pred = np.zeros_like(y)
                y_proba = np.zeros(len(y))

                for train_idx, test_idx in loo.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model = LogisticRegression(C=c_val, class_weight=cw_val,
                                              random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred[test_idx] = model.predict(X_test)
                    y_proba[test_idx] = model.predict_proba(X_test)[:, 1]

                # Evaluation metrics
                accuracy = accuracy_score(y, y_pred)
                cm = confusion_matrix(y, y_pred)

                try:
                    roc_auc = roc_auc_score(y, y_proba)
                except:
                    roc_auc = np.nan

                results_by_config[config_name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba.tolist(),
                }

                print(f"Config: {config_name}")
                print(f"  Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")
                print(f"  Confusion Matrix (rows=True, cols=Pred):")
                print(f"    {cm}")
                print()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config_name

        print(f"{'='*70}")
        print(f"BEST CONFIG: {best_config}")
        print(f"BEST LOOCV ACCURACY: {best_accuracy:.4f}")
        print(f"{'='*70}\n")

        return results_by_config, features_df, best_config, best_accuracy

    def save_results(self, results_by_config, features_df, best_config, best_accuracy):
        """Save results to JSON and CSV files."""

        # Save results summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '1C - Participant-Level Aggregation',
            'num_participants': int(len(features_df)),
            'bipolar_count': int((features_df['label'] == 0).sum()),
            'unipolar_count': int((features_df['label'] == 1).sum()),
            'best_config': best_config,
            'best_loocv_accuracy': float(best_accuracy),
            'all_results': {k: {
                'accuracy': float(v['accuracy']),
                'roc_auc': float(v['roc_auc']) if not np.isnan(v['roc_auc']) else None,
                'cm_tn': int(v['confusion_matrix'][0][0]),
                'cm_fp': int(v['confusion_matrix'][0][1]),
                'cm_fn': int(v['confusion_matrix'][1][0]),
                'cm_tp': int(v['confusion_matrix'][1][1]),
            } for k, v in results_by_config.items()}
        }

        # Save to JSON
        json_path = self.results_dir / "results_1c.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save features CSV
        csv_path = self.results_dir / "participant_features.csv"
        features_df.to_csv(csv_path, index=False)

        print(f"[SAVED] Results saved to {json_path}")
        print(f"[SAVED] Features saved to {csv_path}")

        return summary


def main():
    """Run Approach 1C."""
    classifier = ParticipantVariabilityClassifier()

    # Compute features
    features_df = classifier.compute_participant_features()

    if len(features_df) == 0:
        print("ERROR: No participants loaded!")
        return

    # Test hyperparameters
    c_values = [0.1, 1.0, 10.0]
    class_weights = [None, 'balanced']

    results_by_config, features_df, best_config, best_accuracy = \
        classifier.train_and_evaluate_loocv(features_df, c_values, class_weights)

    # Save results
    summary = classifier.save_results(results_by_config, features_df, best_config, best_accuracy)

    print("\n" + "="*70)
    print("APPROACH 1C COMPLETE")
    print("="*70)
    print(f"Summary: {len(features_df)} participants classified")
    print(f"Best LOOCV Accuracy: {best_accuracy:.2%}")
    print(f"Key finding: Variability does {'[YES]' if best_accuracy > 0.60 else '[NO]'} help detection")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
