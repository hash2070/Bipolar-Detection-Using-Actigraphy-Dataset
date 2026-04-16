"""
CODE TEMPLATES: Approach 3 Implementations

This file contains skeleton code for the top 3 recommended approaches.
Copy and modify as needed. All use existing data_loader.py for consistency.
"""

# ============================================================================
# APPROACH 1C: PARTICIPANT-LEVEL AGGREGATION
# ============================================================================
# File: classify_by_variability.py
# Time: 3-4 hours to implement + test
# Expected: 60-70% LOOCV accuracy

"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from data_loader import DepresjonDataLoader


class ParticipantVariabilityClassifier:
    def __init__(self, data_path: str = "depresjon/data"):
        self.loader = DepresjonDataLoader(data_path)
        self.X, self.y, self.metadata = self.loader.create_experiment_dataset(experiment=2)
        
    def compute_participant_features(self) -> pd.DataFrame:
        \"\"\"
        Compute variability metrics for each participant.
        Returns DataFrame with one row per participant.
        \"\"\"
        participant_ids = []
        labels = []
        features_list = []
        
        # Get unique participants
        unique_pids = self.loader.metadata[
            self.loader.metadata['is_condition']
        ]['participant_id'].values
        
        for pid in unique_pids:
            # Get metadata
            meta = self.loader.metadata[
                self.loader.metadata['participant_id'] == pid
            ].iloc[0]
            label = int(meta['afftype'] == 2)  # 0=bipolar, 1=unipolar
            
            # Load activity data
            activity_data = self.loader._load_activity(pid)
            activity = activity_data['activity'].values.astype(float)
            
            # Compute daily statistics
            daily_stats = []
            for day in range(0, len(activity) - 1440, 1440):
                day_activity = activity[day:day+1440]
                daily_stats.append({
                    'day_mean': day_activity.mean(),
                    'day_std': day_activity.std(),
                    'day_max': day_activity.max(),
                })
            
            daily_means = [d['day_mean'] for d in daily_stats]
            daily_stds = [d['day_std'] for d in daily_stats]
            
            # Feature set
            features = {
                'participant_id': pid,
                'label': label,
                'mean_activity': np.mean(daily_means),
                'variability_across_days': np.std(daily_means),  # KEY FEATURE
                'max_activity': np.max([d['day_max'] for d in daily_stats]),
                'min_activity': np.min(daily_means),
                'range': np.max(daily_means) - np.min(daily_means),
                'coefficient_of_variation': np.std(daily_means) / np.mean(daily_means) if np.mean(daily_means) > 0 else 0,
                'mean_daily_std': np.mean(daily_stds),
                'n_days': len(daily_stats),
            }
            
            participant_ids.append(pid)
            labels.append(label)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train_and_evaluate_loocv(self):
        \"\"\"
        Train logistic regression using Leave-One-Out CV.
        \"\"\"
        features_df = self.compute_participant_features()
        
        # Prepare feature matrix
        feature_cols = ['mean_activity', 'variability_across_days', 'range', 
                       'coefficient_of_variation', 'mean_daily_std']
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        y_pred = np.zeros_like(y)
        y_proba = np.zeros(len(y))
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
            y_proba[test_idx] = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        try:
            roc_auc = roc_auc_score(y, y_proba)
        except:
            roc_auc = np.nan
        
        print(f"\\nParticipant-Level Aggregation Results")
        print(f"LOOCV Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\\n{cm}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Feature importance
        print(f"\\nModel Weights:")
        for i, col in enumerate(feature_cols):
            print(f"  {col}: {model.coef_[0, i]:.4f}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'features_df': features_df,
        }


if __name__ == '__main__':
    classifier = ParticipantVariabilityClassifier()
    results = classifier.train_and_evaluate_loocv()
"""

# ============================================================================
# APPROACH 1A: MULTI-SCALE TEMPORAL WINDOWS
# ============================================================================
# File: train_exp2_multiscale.py
# Time: 4-6 hours + 2-3 hours training
# Expected: 70-75% best accuracy

"""
import torch
import numpy as np
from data_loader import DepresjonDataLoader
from model import CNNLSTMClassifier
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


class MultiScaleExperiment:
    def __init__(self):
        self.window_sizes = {
            '6hr': 360,
            '24hr': 1440,
            '48hr': 2880,
            '72hr': 4320,
            '7day': 10080,
        }
        self.results = {}
        
    def train_model_for_window_size(self, window_minutes: int, window_name: str):
        \"\"\"Train separate model for each window size.\"\"\"
        
        print(f"\\n{'='*60}")
        print(f"Training on {window_name} windows ({window_minutes} minutes)")
        print(f"{'='*60}")
        
        # Load data
        loader = DepresjonDataLoader()
        X, y, metadata = loader.create_experiment_dataset(
            experiment=2,
            window_minutes=window_minutes,
            stride_minutes=int(window_minutes / 2)  # 50% overlap
        )
        
        # Handle participants without enough data
        if len(X) < 100:
            print(f"Warning: Only {len(X)} windows for {window_name}, skipping")
            return None
        
        # Create splits
        split = loader.create_participant_level_split(X, y, metadata)
        
        # Prepare data
        X_train = torch.FloatTensor(split['X_train']).unsqueeze(1)  # (N, 1, L)
        X_val = torch.FloatTensor(split['X_val']).unsqueeze(1)
        X_test = torch.FloatTensor(split['X_test']).unsqueeze(1)
        
        y_train = torch.LongTensor(split['y_train'])
        y_val = torch.LongTensor(split['y_val'])
        y_test = torch.LongTensor(split['y_test'])
        
        # Train
        model = CNNLSTMClassifier(num_classes=2)
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
                torch.save(model.state_dict(), f'best_model_{window_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f'best_model_{window_name}.pt'))
        model.eval()
        
        with torch.no_grad():
            test_logits = model(X_test)
            test_proba = torch.softmax(test_logits, dim=1).numpy()
            test_pred = np.argmax(test_proba, axis=1)
        
        accuracy = accuracy_score(y_test.numpy(), test_pred)
        cm = confusion_matrix(y_test.numpy(), test_pred)
        
        try:
            roc_auc = roc_auc_score(y_test.numpy(), test_proba[:, 1])
        except:
            roc_auc = np.nan
        
        print(f"\\nResults for {window_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Confusion Matrix:\\n{cm}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': test_pred,
            'y_proba': test_proba,
        }
    
    def run_all(self):
        \"\"\"Train models for all window sizes.\"\"\"
        for window_name, window_minutes in self.window_sizes.items():
            result = self.train_model_for_window_size(window_minutes, window_name)
            if result:
                self.results[window_name] = result
        
        # Summary
        print(f"\\n{'='*60}")
        print("SUMMARY: Multi-Scale Results")
        print(f"{'='*60}")
        for window_name, result in self.results.items():
            print(f"{window_name:10s}: Accuracy={result['accuracy']:.4f}, ROC-AUC={result['roc_auc']:.4f}")


if __name__ == '__main__':
    experiment = MultiScaleExperiment()
    experiment.run_all()
"""

# ============================================================================
# APPROACH 1B: FEATURE ENGINEERING + XGBOOST
# ============================================================================
# File: extract_features.py
# Time: 6-8 hours

"""
import numpy as np
import pandas as pd
from data_loader import DepresjonDataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import roc_auc_score, confusion_matrix
import xgboost as xgb


def extract_features_for_participant(activity: np.ndarray, participant_id: str, 
                                    label: int) -> dict:
    \"\"\"Extract 20+ features from participant activity time series.\"\"\"
    
    # Daily decomposition
    daily_activity = []
    for day in range(0, len(activity) - 1440, 1440):
        daily_activity.append(activity[day:day+1440])
    
    daily_means = [d.mean() for d in daily_activity]
    daily_stds = [d.std() for d in daily_activity]
    daily_maxs = [d.max() for d in daily_activity]
    
    # Basic statistics
    features = {
        'participant_id': participant_id,
        'label': label,
        'activity_mean': activity.mean(),
        'activity_std': activity.std(),
        'activity_max': activity.max(),
        'activity_min': activity.min(),
        'activity_iqr': np.percentile(activity, 75) - np.percentile(activity, 25),
    }
    
    # Temporal dynamics
    features['day_to_day_variability'] = np.std(daily_means)
    features['day_to_day_max_range'] = np.max(daily_means) - np.min(daily_means)
    features['coefficient_of_variation'] = features['day_to_day_variability'] / (features['activity_mean'] + 1e-8)
    
    # Sleep/activity fragmentation
    zero_periods = np.sum(activity < np.percentile(activity, 10))
    features['low_activity_percentage'] = zero_periods / len(activity)
    
    # Autocorrelation at key lags
    def autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
    
    features['autocorr_24hr'] = autocorr(daily_means, 1)
    features['autocorr_48hr'] = autocorr(daily_means, 2)
    
    # Entropy (disorder/predictability)
    hist, _ = np.histogram(activity, bins=30)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist/hist.sum() * np.log(hist/hist.sum()))
    
    # Trend analysis
    if len(daily_means) > 1:
        trend = np.polyfit(np.arange(len(daily_means)), daily_means, 1)[0]
        features['trend'] = trend
    else:
        features['trend'] = 0
    
    return features


def train_xgboost_classifier():
    \"\"\"Extract features and train XGBoost for bipolar vs unipolar.\"\"\"
    
    loader = DepresjonDataLoader()
    
    # Extract features for each condition participant
    all_features = []
    unique_pids = loader.metadata[
        loader.metadata['is_condition']
    ]['participant_id'].values
    
    for pid in unique_pids:
        meta = loader.metadata[
            loader.metadata['participant_id'] == pid
        ].iloc[0]
        label = int(meta['afftype'] == 2)
        
        activity_data = loader._load_activity(pid)
        activity = activity_data['activity'].values.astype(float)
        
        features = extract_features_for_participant(activity, pid, label)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Prepare for modeling
    feature_cols = [c for c in features_df.columns if c not in ['participant_id', 'label']]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Train XGBoost with LOOCV
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    y_proba = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        y_proba[test_idx] = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    
    print(f"XGBoost LOOCV Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\\n{cm}")
    
    # Feature importance
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\nTop Features:")
    print(importance.head(10))


if __name__ == '__main__':
    train_xgboost_classifier()
"""

# ============================================================================
# APPROACH 3A: STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
# File: statistical_tests.py
# Time: 2-3 hours

"""
import numpy as np
from scipy.stats import ttest_ind
from data_loader import DepresjonDataLoader


def statistical_analysis():
    \"\"\"Test if bipolar/unipolar differ statistically.\"\"\"
    
    loader = DepresjonDataLoader()
    
    bipolar_activity = []
    unipolar_activity = []
    
    # Collect activity for each group
    unique_pids = loader.metadata[
        loader.metadata['is_condition']
    ]['participant_id'].values
    
    for pid in unique_pids:
        meta = loader.metadata[
            loader.metadata['participant_id'] == pid
        ].iloc[0]
        
        activity_data = loader._load_activity(pid)
        activity = activity_data['activity'].values.astype(float)
        daily_means = [activity[i:i+1440].mean() for i in range(0, len(activity)-1440, 1440)]
        activity_variability = np.std(daily_means)
        
        if meta['afftype'] == 1:  # Bipolar
            bipolar_activity.append(activity_variability)
        else:  # Unipolar
            unipolar_activity.append(activity_variability)
    
    bipolar_activity = np.array(bipolar_activity)
    unipolar_activity = np.array(unipolar_activity)
    
    # T-test
    t_stat, p_value = ttest_ind(bipolar_activity, unipolar_activity)
    
    # Effect size (Cohen's d)
    cohens_d = (bipolar_activity.mean() - unipolar_activity.mean()) / np.sqrt(
        ((len(bipolar_activity)-1)*bipolar_activity.std()**2 + 
         (len(unipolar_activity)-1)*unipolar_activity.std()**2) / 
        (len(bipolar_activity) + len(unipolar_activity) - 2)
    )
    
    print(f"Statistical Significance Test: Activity Variability")
    print(f"Bipolar (n={len(bipolar_activity)}): mean={bipolar_activity.mean():.4f}, std={bipolar_activity.std():.4f}")
    print(f"Unipolar (n={len(unipolar_activity)}): mean={unipolar_activity.mean():.4f}, std={unipolar_activity.std():.4f}")
    print(f"\\nT-test: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    if p_value < 0.05:
        print(f"✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"✗ NOT statistically significant (p ≥ 0.05)")
    
    if abs(cohens_d) < 0.2:
        print(f"  → Effect size: VERY SMALL (d={cohens_d:.4f})")
    elif abs(cohens_d) < 0.5:
        print(f"  → Effect size: SMALL (d={cohens_d:.4f})")
    elif abs(cohens_d) < 0.8:
        print(f"  → Effect size: MEDIUM (d={cohens_d:.4f})")
    else:
        print(f"  → Effect size: LARGE (d={cohens_d:.4f})")


if __name__ == '__main__':
    statistical_analysis()
"""

# ============================================================================
# NOTES
# ============================================================================
"""
USAGE:
1. Copy one of the above code blocks into a new .py file in the project directory
2. Ensure data_loader.py is in the same directory
3. Run: python filename.py

DEPENDENCIES:
- data_loader.py (existing)
- torch, sklearn, numpy, pandas, scipy, xgboost (should be installed)

CUSTOMIZATION:
- Modify feature_cols, window_sizes, hyperparameters as needed
- Change window stride (currently 50% overlap, can try 75%)
- Add more features in extract_features_for_participant()
- Try different LOOCV or K-fold cross-validation

EXPECTED TIMELINE:
- 1C: 3-4 hours
- 1A: 4-6 hours (+ 2-3 hours training)
- 1B: 6-8 hours
- 3A: 2-3 hours
"""
