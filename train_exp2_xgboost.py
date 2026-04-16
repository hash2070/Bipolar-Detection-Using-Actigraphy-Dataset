"""
APPROACH 1B: Feature Engineering + XGBoost
Extract hand-crafted features from activity data and train XGBoost classifier.
Uses LOOCV for validation on 23 mood-disorder participants.

Key insight: Prior work used engineered features and achieved F1=0.82
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from data_loader import DepresjonDataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb


def extract_features_for_activity(activity: np.ndarray, participant_id: str, label: int):
    """
    Extract 20+ engineered features from participant activity time series.
    These features capture temporal dynamics, variability, and patterns.
    """

    # Normalize
    activity = (activity - activity.mean()) / (activity.std() + 1e-8)

    # Daily decomposition (1440 minutes = 24 hours)
    daily_activity = []
    for day in range(0, len(activity) - 1440, 1440):
        daily_activity.append(activity[day:day+1440])

    if len(daily_activity) < 2:
        return None

    daily_means = np.array([d.mean() for d in daily_activity])
    daily_stds = np.array([d.std() for d in daily_activity])
    daily_maxs = np.array([d.max() for d in daily_activity])
    daily_mins = np.array([d.min() for d in daily_activity])

    features = {
        'participant_id': participant_id,
        'label': label,
        'num_days': len(daily_activity),
    }

    # 1. Basic statistics
    features['activity_mean'] = activity.mean()
    features['activity_std'] = activity.std()
    features['activity_max'] = activity.max()
    features['activity_min'] = activity.min()
    features['activity_iqr'] = np.percentile(activity, 75) - np.percentile(activity, 25)

    # 2. Day-to-day variability (KEY for bipolar detection)
    features['day_variability'] = daily_means.std()
    features['day_range'] = daily_means.max() - daily_means.min()
    features['coefficient_of_variation'] = features['day_variability'] / (np.abs(features['activity_mean']) + 1e-8)

    # 3. Within-day stability
    features['mean_daily_std'] = daily_stds.mean()
    features['mean_daily_max'] = daily_maxs.mean()
    features['daily_max_variability'] = daily_maxs.std()

    # 4. Activity fragmentation
    low_activity_threshold = np.percentile(activity, 25)
    features['low_activity_fraction'] = (activity < low_activity_threshold).sum() / len(activity)

    high_activity_threshold = np.percentile(activity, 75)
    features['high_activity_fraction'] = (activity > high_activity_threshold).sum() / len(activity)

    # 5. Autocorrelation (temporal dependencies)
    def lag_autocorr(x, lag):
        if len(x) > lag:
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return 0

    features['autocorr_lag1'] = lag_autocorr(daily_means, 1)
    features['autocorr_lag2'] = lag_autocorr(daily_means, 2) if len(daily_means) > 2 else 0

    # 6. Trend analysis
    if len(daily_means) > 1:
        x = np.arange(len(daily_means))
        trend_coef = np.polyfit(x, daily_means, 1)[0]
        features['trend'] = trend_coef
    else:
        features['trend'] = 0

    # 7. Entropy (predictability/disorder)
    hist, _ = np.histogram(activity, bins=30)
    hist = hist[hist > 0]
    if len(hist) > 0:
        features['entropy'] = -np.sum(hist/hist.sum() * np.log(hist/hist.sum()))
    else:
        features['entropy'] = 0

    # 8. Peak characteristics (2-hour intervals to avoid reshape issues)
    if len(activity) >= 120:  # At least 2 hours
        interval_means = []
        for i in range(0, len(activity) - 120, 120):
            interval_means.append(activity[i:i+120].mean())
        if len(interval_means) > 2:
            features['num_peaks'] = len([j for j in range(1, len(interval_means)-1)
                                         if interval_means[j] > interval_means[j-1]
                                         and interval_means[j] > interval_means[j+1]])
        else:
            features['num_peaks'] = 0
    else:
        features['num_peaks'] = 0

    # 9. Sleep/wake cycle regularity
    zero_sequences = 0
    in_sequence = False
    for val in activity:
        if val < low_activity_threshold:
            if not in_sequence:
                zero_sequences += 1
                in_sequence = True
        else:
            in_sequence = False
    features['sleep_cycles'] = zero_sequences

    return features


class FeatureEngineeringClassifier:
    """Extract features and train XGBoost for bipolar vs unipolar."""

    def __init__(self):
        self.results_dir = Path("findings/1B")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_features(self):
        """Extract features for all condition participants."""

        loader = DepresjonDataLoader()
        all_features = []

        unique_pids = loader.metadata[
            loader.metadata['is_condition']
        ]['participant_id'].unique()

        print(f"\nExtracting features for {len(unique_pids)} participants...")

        for pid in unique_pids:
            meta = loader.metadata[
                loader.metadata['participant_id'] == pid
            ].iloc[0]

            label = int(meta['afftype'] == 2)  # 0=bipolar, 1=unipolar
            afftype_name = "Unipolar" if label == 1 else "Bipolar"

            # Load activity
            activity_data = loader._load_activity(pid)
            activity = activity_data['activity'].values.astype(float)

            # Extract features
            features = extract_features_for_activity(activity, pid, label)
            if features:
                all_features.append(features)
                print(f"  [OK] {pid:15s} ({afftype_name:8s}): {features['num_days']} days")

        return pd.DataFrame(all_features)

    def train_and_evaluate_loocv(self, features_df: pd.DataFrame, max_depth=5, n_estimators=100):
        """Train XGBoost using Leave-One-Out CV."""

        feature_cols = [c for c in features_df.columns if c not in ['participant_id', 'label', 'num_days']]
        X = features_df[feature_cols].values
        y = features_df['label'].values

        print(f"\n{'='*70}")
        print(f"APPROACH 1B: XGBoost LOOCV")
        print(f"{'='*70}")
        print(f"Participants: {len(y)} (Bipolar: {(y==0).sum()}, Unipolar: {(y==1).sum()})")
        print(f"Features: {len(feature_cols)}")
        print(f"XGBoost params: max_depth={max_depth}, n_estimators={n_estimators}")
        print(f"\n{'-'*70}\n")

        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        y_pred = np.zeros_like(y)
        y_proba = np.zeros(len(y))
        feature_importances = np.zeros(len(feature_cols))

        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = xgb.XGBClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
            y_proba[test_idx] = model.predict_proba(X_test)[:, 1]
            feature_importances += model.feature_importances_

            if (fold + 1) % 5 == 0:
                print(f"  Completed {fold+1}/{len(y)} folds")

        # Average feature importances
        feature_importances /= len(y)

        # Evaluation
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        try:
            roc_auc = roc_auc_score(y, y_proba)
        except:
            roc_auc = np.nan

        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"{'='*70}\n")

        # Feature importance ranking
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        print(f"Top 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
            'feature_importance': importance_df.to_dict()
        }, features_df

    def save_results(self, results, features_df, accuracy):
        """Save results to JSON and CSV."""

        summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '1B - Feature Engineering + XGBoost',
            'num_participants': int(len(features_df)),
            'bipolar_count': int((features_df['label'] == 0).sum()),
            'unipolar_count': int((features_df['label'] == 1).sum()),
            'loocv_accuracy': float(accuracy),
            'roc_auc': float(results['roc_auc']) if not np.isnan(results['roc_auc']) else None,
            'confusion_matrix': results['confusion_matrix'],
            'top_features': list(results['feature_importance']['feature'].values())[:10],
        }

        # Save summary
        json_path = self.results_dir / "results_1b.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save feature importance
        features_csv = self.results_dir / "feature_importance.csv"
        feats_df = pd.DataFrame({
            'feature': results['feature_importance']['feature'],
            'importance': results['feature_importance']['importance']
        })
        feats_df.to_csv(features_csv, index=False)

        return summary


if __name__ == '__main__':
    classifier = FeatureEngineeringClassifier()

    # Extract features (once, reused for all configs)
    features_df = classifier.extract_all_features()

    if len(features_df) == 0:
        print("ERROR: No features extracted!")
    else:
        # --- Hyperparameter Grid Search: max_depth=[3,5,7], n_estimators fixed at 100 ---
        max_depths = [3, 5, 7]
        n_estimators = 100  # SKIP n_estimators tuning (low ROI per analysis)

        grid_results = {}
        best_accuracy = -1
        best_config = None
        best_results = None

        print("\n" + "="*70)
        print("APPROACH 1B: Hyperparameter Grid Search")
        print(f"max_depth values: {max_depths}  |  n_estimators: {n_estimators} (fixed)")
        print("="*70)

        for max_depth in max_depths:
            print(f"\n--- Testing max_depth={max_depth} ---")
            results, _ = classifier.train_and_evaluate_loocv(
                features_df, max_depth=max_depth, n_estimators=n_estimators
            )
            grid_results[f'max_depth={max_depth}'] = {
                'accuracy': float(results['accuracy']),
                'roc_auc': float(results['roc_auc']) if not (isinstance(results['roc_auc'], float) and
                            results['roc_auc'] != results['roc_auc']) else None,
                'confusion_matrix': results['confusion_matrix'],
            }
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_config = f'max_depth={max_depth}'
                best_results = results

        # Print grid search summary
        print("\n" + "="*70)
        print("GRID SEARCH SUMMARY:")
        print("="*70)
        for cfg, res in grid_results.items():
            print(f"  {cfg:20s} -> accuracy={res['accuracy']:.4f}, roc_auc={res['roc_auc']}")
        print(f"\nBest config: {best_config} (accuracy={best_accuracy:.4f})")

        # Save best + grid results
        summary = classifier.save_results(best_results, features_df, best_accuracy)

        # Also save full grid results
        import json
        from datetime import datetime
        grid_summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '1B - Feature Engineering + XGBoost (Grid Search)',
            'num_participants': int(len(features_df)),
            'bipolar_count': int((features_df['label'] == 0).sum()),
            'unipolar_count': int((features_df['label'] == 1).sum()),
            'grid_search': {
                'max_depths_tested': max_depths,
                'n_estimators_fixed': n_estimators,
            },
            'all_results': grid_results,
            'best_config': best_config,
            'best_accuracy': float(best_accuracy),
        }
        with open("findings/1B/results_1b_gridsearch.json", 'w') as f:
            json.dump(grid_summary, f, indent=2)
        print(f"\n[SAVED] Grid search results -> findings/1B/results_1b_gridsearch.json")

        print("\n" + "="*70)
        print("APPROACH 1B GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Participants: {len(features_df)}")
        print(f"Best Config: {best_config}")
        print(f"Best LOOCV Accuracy: {best_accuracy:.2%}")
        print("="*70 + "\n")
