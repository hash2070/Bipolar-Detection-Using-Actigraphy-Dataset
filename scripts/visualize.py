"""
Visualization utilities for experiment results.
Generates ROC curves, confusion matrices, and other diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path


def plot_roc_curve(y_true, y_probs, experiment_num, save_path="results"):
    """Plot ROC curve for binary classification."""
    Path(save_path).mkdir(exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Experiment {experiment_num}: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/exp{experiment_num}_roc_curve.png', dpi=150)
    print(f"Saved ROC curve to {save_path}/exp{experiment_num}_roc_curve.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, experiment_num, labels=None, save_path="results"):
    """Plot confusion matrix heatmap."""
    Path(save_path).mkdir(exist_ok=True)

    if labels is None:
        if experiment_num == 1:
            labels = ['Healthy', 'Depressed']
        else:
            labels = ['Bipolar', 'Unipolar']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Experiment {experiment_num}: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_path}/exp{experiment_num}_confusion_matrix.png', dpi=150)
    print(f"Saved confusion matrix to {save_path}/exp{experiment_num}_confusion_matrix.png")
    plt.close()


def plot_training_history(history, experiment_num, save_path="results"):
    """Plot training and validation loss/accuracy curves."""
    Path(save_path).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Experiment {experiment_num}: Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Experiment {experiment_num}: Training Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/exp{experiment_num}_training_history.png', dpi=150)
    print(f"Saved training history to {save_path}/exp{experiment_num}_training_history.png")
    plt.close()


def generate_all_visualizations(experiment_num, save_path="results"):
    """Generate all visualizations for an experiment."""
    print(f"\nGenerating visualizations for Experiment {experiment_num}...")

    # Load results
    y_true = np.load(f'{save_path}/exp{experiment_num}_y_true.npy')
    y_pred = np.load(f'{save_path}/exp{experiment_num}_y_pred.npy')
    y_probs = np.load(f'{save_path}/exp{experiment_num}_y_probs.npy')

    # Generate plots
    plot_roc_curve(y_true, y_probs, experiment_num, save_path)
    plot_confusion_matrix(y_true, y_pred, experiment_num, save_path=save_path)

    print(f"Visualizations for Experiment {experiment_num} complete!")


if __name__ == "__main__":
    # After experiments complete, run:
    # python visualize.py
    print("Import this module and call generate_all_visualizations(1) or generate_all_visualizations(2)")
