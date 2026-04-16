"""
Data loading and preprocessing for Depresjon actigraphy dataset.
Handles 1D-CNN-LSTM input preparation with participant-level stratification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split


class DepresjonDataLoader:
    """Load and preprocess Depresjon actigraphy dataset from extracted folder."""

    def __init__(self, data_path: str = "depresjon/data"):
        """Initialize with path to extracted depresjon/data directory."""
        self.data_path = Path(data_path)
        self.metadata = None
        self.activity_data = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load scores.csv with participant metadata."""
        scores_path = self.data_path / 'scores.csv'
        self.metadata = pd.read_csv(scores_path)

        # Parse afftype: 1=Bipolar II, 2=Unipolar, NA=Bipolar I
        # For experiments: condition=1 (any mood disorder), healthy=0 (controls)
        self.metadata['participant_id'] = self.metadata['number']
        self.metadata['is_condition'] = ~self.metadata['afftype'].isna()
        self.metadata['afftype'] = self.metadata['afftype'].fillna(1.5)  # Bipolar I placeholder

        print(f"Loaded metadata for {len(self.metadata)} participants:")
        print(f"  Controls: {(~self.metadata['is_condition']).sum()}")
        print(f"  Condition (Unipolar=2, Bipolar I~1.5, Bipolar II=1): {self.metadata['is_condition'].sum()}")

    def _load_activity(self, participant_id: str) -> pd.DataFrame:
        """Load activity CSV for a participant from extracted folder."""
        if participant_id in self.activity_data:
            return self.activity_data[participant_id]

        # Extract participant type and number from metadata
        row = self.metadata[self.metadata['participant_id'] == participant_id].iloc[0]
        is_condition = row['is_condition']

        prefix = 'condition' if is_condition else 'control'
        num = participant_id.split('_')[1]
        csv_path = self.data_path / prefix / f'{prefix}_{num}.csv'

        df = pd.read_csv(csv_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.activity_data[participant_id] = df
        return df

    def preprocess_participant(self, participant_id: str,
                               window_minutes: int = 1440,
                               stride_minutes: int = 60) -> Tuple[np.ndarray, List[Dict]]:
        """
        Preprocess activity data for a participant.

        Args:
            participant_id: 'condition_X' or 'control_X'
            window_minutes: 1440 = 24 hours
            stride_minutes: window stride during training

        Returns:
            windows: (num_windows, window_minutes) activity array
            metadata_list: list of window metadata dicts
        """
        df = self._load_activity(participant_id)
        meta_row = self.metadata[self.metadata['participant_id'] == participant_id].iloc[0]

        # Standardize to zero mean, unit variance
        activity = df['activity'].values.astype(float)
        activity = (activity - activity.mean()) / (activity.std() + 1e-8)

        windows = []
        window_metas = []

        # Slide window across the time series
        for start_idx in range(0, len(activity) - window_minutes + 1, stride_minutes):
            end_idx = start_idx + window_minutes
            window = activity[start_idx:end_idx]

            windows.append(window)
            window_metas.append({
                'participant_id': participant_id,
                'is_condition': meta_row['is_condition'],
                'afftype': meta_row['afftype'],  # 2=unipolar, 1=bipolar II, 1.5=bipolar I
                'window_start_idx': start_idx,
                'madrs_start': meta_row['madrs1'],
                'madrs_end': meta_row['madrs2'],
            })

        return np.array(windows), window_metas

    def create_experiment_dataset(self, experiment: int = 1,
                                   window_minutes: int = 1440,
                                   stride_minutes: int = 60) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create dataset for Experiment 1 or 2.

        Args:
            experiment: 1 for healthy vs depressed, 2 for bipolar vs unipolar
            window_minutes: 1440 = 24 hours
            stride_minutes: window stride

        Returns:
            X: (num_samples, window_minutes) activity windows
            y: (num_samples,) binary labels
            metadata: list of sample metadata dicts
        """
        all_windows = []
        all_labels = []
        all_metadata = []

        for _, row in self.metadata.iterrows():
            participant_id = row['participant_id']

            # Filter participants based on experiment
            if experiment == 1:
                # Healthy vs. Depressed: include all 55 participants
                include = True
                label = int(row['is_condition'])  # 0=healthy, 1=depressed
            elif experiment == 2:
                # Bipolar vs. Unipolar: only condition participants
                include = row['is_condition']
                if include:
                    label = int(row['afftype'] == 2)  # 0=bipolar, 1=unipolar
            else:
                raise ValueError("experiment must be 1 or 2")

            if not include:
                continue

            try:
                windows, window_metas = self.preprocess_participant(
                    participant_id, window_minutes, stride_minutes
                )

                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                all_metadata.extend(window_metas)

            except Exception as e:
                print(f"Error processing {participant_id}: {e}")

        X = np.concatenate(all_windows, axis=0)
        y = np.array(all_labels)

        print(f"\nExperiment {experiment} dataset created:")
        print(f"  Total windows: {len(X)}")
        print(f"  Label distribution: {np.bincount(y)}")
        print(f"  Shape: {X.shape}")

        return X, y, all_metadata

    def create_participant_level_split(self, X: np.ndarray, y: np.ndarray,
                                       metadata: List[Dict],
                                       train_size: float = 0.8,
                                       val_size: float = 0.1,
                                       test_size: float = 0.1,
                                       random_state: int = 42) -> Dict:
        """
        Create train/val/test splits at participant level to prevent data leakage.

        Returns:
            dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Get unique participants and their labels
        participant_ids = np.array([m['participant_id'] for m in metadata])
        participant_labels = np.array([m['is_condition'] for m in metadata])

        unique_participants, first_indices = np.unique(participant_ids, return_index=True)
        participant_classes = participant_labels[first_indices]

        # First split: 80% train, 20% temp (for val+test)
        train_pids, temp_pids, train_classes, temp_classes = train_test_split(
            unique_participants, participant_classes,
            train_size=train_size,
            stratify=participant_classes,
            random_state=random_state
        )

        # Second split: split remaining 20% into 50% val, 50% test
        val_size_ratio = val_size / (val_size + test_size)
        val_pids, test_pids, _, _ = train_test_split(
            temp_pids, temp_classes,
            train_size=val_size_ratio,
            stratify=temp_classes,
            random_state=random_state
        )

        # Create masks for samples
        train_mask = np.isin(participant_ids, train_pids)
        val_mask = np.isin(participant_ids, val_pids)
        test_mask = np.isin(participant_ids, test_pids)

        result = {
            'X_train': X[train_mask],
            'X_val': X[val_mask],
            'X_test': X[test_mask],
            'y_train': y[train_mask],
            'y_val': y[val_mask],
            'y_test': y[test_mask],
            'train_participants': train_pids,
            'val_participants': val_pids,
            'test_participants': test_pids,
        }

        print(f"\nParticipant-level split:")
        print(f"  Train: {len(train_pids)} participants, {result['X_train'].shape[0]} windows")
        print(f"  Val:   {len(val_pids)} participants, {result['X_val'].shape[0]} windows")
        print(f"  Test:  {len(test_pids)} participants, {result['X_test'].shape[0]} windows")

        return result


if __name__ == "__main__":
    loader = DepresjonDataLoader("depresjon/data")

    # Test Experiment 1
    X_exp1, y_exp1, meta_exp1 = loader.create_experiment_dataset(experiment=1)
    split_exp1 = loader.create_participant_level_split(X_exp1, y_exp1, meta_exp1)

    print(f"\nExp 1 label distribution in train set: {np.bincount(split_exp1['y_train'])}")
