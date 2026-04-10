"""
Visualization utilities for generating poster figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path


def plot_roc_auc(y_true: np.ndarray, y_probs: np.ndarray,
                 experiment: int, save_path: str = "results") -> None:
    """Plot ROC-AUC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC-AUC Curve - Experiment {experiment}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    Path(save_path).mkdir(exist_ok=True)
    plt.savefig(f"{save_path}/roc_auc_exp{experiment}.png", dpi=300, bbox_inches='tight')
    print(f"Saved ROC-AUC plot: {save_path}/roc_auc_exp{experiment}.png")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          experiment: int, labels: list = None,
                          save_path: str = "results") -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    if labels is None:
        labels = ['Class 0', 'Class 1']
    if experiment == 1:
        labels = ['Healthy', 'Depressed']
    elif experiment == 2:
        labels = ['Bipolar', 'Unipolar']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - Experiment {experiment}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path(save_path).mkdir(exist_ok=True)
    plt.savefig(f"{save_path}/confusion_matrix_exp{experiment}.png", dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {save_path}/confusion_matrix_exp{experiment}.png")
    plt.close()


def plot_training_history(history: dict, experiment: int, save_path: str = "results") -> None:
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'Experiment {experiment} - Training History', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    Path(save_path).mkdir(exist_ok=True)
    plt.savefig(f"{save_path}/training_history_exp{experiment}.png", dpi=300, bbox_inches='tight')
    print(f"Saved training history: {save_path}/training_history_exp{experiment}.png")
    plt.close()


def plot_sample_actigraphy(X: np.ndarray, y: np.ndarray, num_samples: int = 4,
                           experiment: int = 1, save_path: str = "results") -> None:
    """Plot sample actigraphy 24-hour windows."""
    classes = ['Healthy', 'Depressed'] if experiment == 1 else ['Bipolar', 'Unipolar']

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 6))
    axes = axes.flatten()

    for class_idx in range(2):
        class_indices = np.where(y == class_idx)[0]
        for i in range(num_samples // 2):
            if i < len(class_indices):
                ax = axes[class_idx * (num_samples // 2) + i]
                ax.plot(X[class_indices[i]], linewidth=1, color='steelblue')
                ax.set_title(f'{classes[class_idx]} - Sample {i+1}', fontsize=11)
                ax.set_xlabel('Minutes (24h window)', fontsize=10)
                ax.set_ylabel('Activity (normalized)', fontsize=10)
                ax.grid(alpha=0.3)

    plt.suptitle(f'Sample Actigraphy Windows - Experiment {experiment}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    Path(save_path).mkdir(exist_ok=True)
    plt.savefig(f"{save_path}/sample_actigraphy_exp{experiment}.png", dpi=300, bbox_inches='tight')
    print(f"Saved sample actigraphy: {save_path}/sample_actigraphy_exp{experiment}.png")
    plt.close()


def generate_all_visualizations(exp1_results: dict, exp1_y_true: np.ndarray, exp1_y_pred: np.ndarray,
                                exp1_y_probs: np.ndarray, exp1_history: dict,
                                exp2_results: dict, exp2_y_true: np.ndarray, exp2_y_pred: np.ndarray,
                                exp2_y_probs: np.ndarray, exp2_history: dict,
                                X_exp1: np.ndarray, y_exp1_full: np.ndarray,
                                X_exp2: np.ndarray, y_exp2_full: np.ndarray) -> None:
    """Generate all visualizations for both experiments."""
    print("\nGenerating visualizations for poster...")

    # Experiment 1
    plot_roc_auc(exp1_y_true, exp1_y_probs, experiment=1)
    plot_confusion_matrix(exp1_y_true, exp1_y_pred, experiment=1)
    plot_training_history(exp1_history, experiment=1)
    plot_sample_actigraphy(X_exp1, y_exp1_full, experiment=1)

    # Experiment 2
    plot_roc_auc(exp2_y_true, exp2_y_probs, experiment=2)
    plot_confusion_matrix(exp2_y_true, exp2_y_pred, experiment=2)
    plot_training_history(exp2_history, experiment=2)
    plot_sample_actigraphy(X_exp2, y_exp2_full, experiment=2)

    print("All visualizations complete!")
