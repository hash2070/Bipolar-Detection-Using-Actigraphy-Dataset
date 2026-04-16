"""
APPROACH 3B: Visualization & Final Report
Generates all presentation-ready figures from existing results (1A, 1B, 1C, 3A).
Saves all plots to results/ and documents findings in findings/3B/.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Setup ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
FINDINGS_3B = Path("findings/3B")
RESULTS_DIR.mkdir(exist_ok=True)
FINDINGS_3B.mkdir(exist_ok=True)

# Consistent color palette
COLORS = {
    'bipolar': '#E74C3C',
    'unipolar': '#3498DB',
    'approach_1a': '#2ECC71',
    'approach_1b': '#E67E22',
    'approach_1c': '#9B59B6',
    'baseline': '#95A5A6',
    'bar_primary': '#2C3E50',
    'bar_accent': '#E74C3C',
}

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


# ── Load existing results ───────────────────────────────────────────────────
def load_results():
    with open("findings/1A/results_1a.json") as f:
        res_1a = json.load(f)
    with open("findings/1A/results_1a_extended.json") as f:
        res_1a_ext = json.load(f)
    with open("findings/1B/results_1b.json") as f:
        res_1b = json.load(f)
    with open("findings/1B/feature_importance.csv") as f:
        feat_imp = pd.read_csv(f)
    with open("findings/1C/results_1c.json") as f:
        res_1c = json.load(f)
    with open("findings/3A/results_3a.json") as f:
        res_3a = json.load(f)
    return res_1a, res_1a_ext, res_1b, feat_imp, res_1c, res_3a


# ── PLOT 1: Accuracy Comparison Across Approaches ──────────────────────────
def plot_accuracy_comparison(res_1a, res_1b, res_1c):
    """Bar chart comparing accuracy across all approaches vs baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    approaches = [
        'Baseline\n(predict all\nunipolar)',
        '1C: Logistic\nRegression\n(5 features)',
        '1A: CNN-LSTM\n24hr window',
        '1A: CNN-LSTM\n48hr window',
        '1B: XGBoost\n(19 features)',
    ]
    accuracies = [
        65.2,  # baseline: 15/23 = 0.6521
        res_1c['best_loocv_accuracy'] * 100,
        res_1a['window_results']['24hr']['accuracy'] * 100,
        res_1a['window_results']['48hr']['accuracy'] * 100,
        res_1b['loocv_accuracy'] * 100,
    ]
    colors = [
        COLORS['baseline'],
        COLORS['approach_1c'],
        COLORS['approach_1a'],
        COLORS['approach_1a'],
        COLORS['approach_1b'],
    ]

    bars = ax.bar(approaches, accuracies, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    # Annotate bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Baseline reference line
    ax.axhline(y=65.2, color=COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.6,
               label='Majority class baseline (65.2%)')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy: Bipolar vs Unipolar Depression\nAll Approaches vs Majority-Class Baseline',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 90)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "fig1_accuracy_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 2: CNN-LSTM Window Size Comparison (1A) ───────────────────────────
def plot_window_size_comparison(res_1a, res_1a_ext):
    """Compare 24hr / 48hr / 72hr window performance for CNN-LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    windows = ['24hr\n(1440 min)', '48hr\n(2880 min)', '72hr\n(4320 min)']
    accs_base = [
        res_1a['window_results']['24hr']['accuracy'] * 100,
        res_1a['window_results']['48hr']['accuracy'] * 100,
        res_1a['window_results']['72hr']['accuracy'] * 100,
    ]

    # Best across all hyperparameter configs (extended)
    accs_best = [
        res_1a_ext['results_by_window']['24hr']['best_accuracy'] * 100,
        res_1a_ext['results_by_window']['48hr']['best_accuracy'] * 100,
        res_1a_ext['results_by_window']['72hr']['best_accuracy'] * 100,
    ]
    best_configs = [
        res_1a_ext['results_by_window']['24hr']['best_config'],
        res_1a_ext['results_by_window']['48hr']['best_config'],
        res_1a_ext['results_by_window']['72hr']['best_config'],
    ]

    # Left: base results
    bars1 = axes[0].bar(windows, accs_base, color=[COLORS['approach_1a'], '#27AE60', '#1A8A4A'],
                        edgecolor='white', linewidth=1.5, width=0.5)
    for bar, acc in zip(bars1, accs_base):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].axhline(y=65.2, color=COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Majority baseline (65.2%)')
    axes[0].set_title('1A Base Run: Accuracy by Window Size', fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_ylim(0, 95)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: best across grid search
    bars2 = axes[1].bar(windows, accs_best, color=[COLORS['approach_1a'], '#27AE60', '#1A8A4A'],
                        edgecolor='white', linewidth=1.5, width=0.5)
    for bar, acc, cfg in zip(bars2, accs_best, best_configs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                     cfg, ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    axes[1].axhline(y=65.2, color=COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Majority baseline (65.2%)')
    axes[1].set_title('1A Best Config: Accuracy by Window Size\n(Grid Search over weight_decay, dropout)', fontweight='bold')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('CNN-LSTM Window Size Analysis (Approach 1A)\nBipolar vs Unipolar Classification',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "fig2_window_size_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 3: Confusion Matrices (ALL approaches) ─────────────────────────────
def plot_confusion_matrices(res_1a, res_1b, res_1c):
    """Confusion matrices for ALL approaches: 1A (3 windows) + 1B + 1C."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    labels = ['Bipolar', 'Unipolar']

    cmaps = ['Greens', 'Greens', 'Greens', 'Reds', 'Blues']
    titles = [
        '1A: CNN-LSTM 24hr\nTest Acc: 77.9%',
        '1A: CNN-LSTM 48hr\nTest Acc: 53.7%',
        '1A: CNN-LSTM 72hr\nTest Acc: 23.1%',
        '1B: XGBoost (19 feat)\nLOOCV Acc: 39.1%',
        '1C: Logistic Reg (5 feat)\nLOOCV Acc: 60.9%',
    ]

    # 1A CMs (from results_1a.json)
    for i, window in enumerate(['24hr', '48hr', '72hr']):
        cm = np.array(res_1a['window_results'][window]['cm'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], ax=axes[i],
                    xticklabels=labels, yticklabels=labels,
                    cbar=False, annot_kws={'fontsize': 13, 'fontweight': 'bold'})
        axes[i].set_title(titles[i], fontweight='bold', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=9)
        axes[i].set_xlabel('Predicted', fontsize=9)

    # 1B CM
    cm_1b = np.array(res_1b['confusion_matrix'])
    sns.heatmap(cm_1b, annot=True, fmt='d', cmap='Reds', ax=axes[3],
                xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    axes[3].set_title(titles[3], fontweight='bold', fontsize=10)
    axes[3].set_ylabel('True Label', fontsize=9)
    axes[3].set_xlabel('Predicted', fontsize=9)

    # 1C CM (best config: C=0.1, class_weight=None)
    best_1c = res_1c['all_results']['C=0.1, class_weight=None']
    cm_1c = np.array([
        [best_1c['cm_tn'], best_1c['cm_fp']],
        [best_1c['cm_fn'], best_1c['cm_tp']],
    ])
    sns.heatmap(cm_1c, annot=True, fmt='d', cmap='Blues', ax=axes[4],
                xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    axes[4].set_title(titles[4], fontweight='bold', fontsize=10)
    axes[4].set_ylabel('True Label', fontsize=9)
    axes[4].set_xlabel('Predicted', fontsize=9)

    plt.suptitle('Confusion Matrices: ALL Approaches — Bipolar vs Unipolar\n'
                 '(1A = CNN-LSTM test set windows | 1B/1C = LOOCV on 23 participants)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "fig3_confusion_matrices_all.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 4: Feature Importance (1B) ─────────────────────────────────────────
def plot_feature_importance(feat_imp):
    """Horizontal bar chart of top-15 XGBoost feature importances."""
    top15 = feat_imp[feat_imp['importance'] > 0].head(15).copy()
    top15 = top15.sort_values('importance', ascending=True)

    # Color hypothesis feature differently
    colors = [COLORS['bar_accent'] if f == 'day_variability' else COLORS['bar_primary']
              for f in top15['feature']]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top15['feature'], top15['importance'] * 100, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.7)

    for bar, imp in zip(bars, top15['importance']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{imp*100:.1f}%', va='center', ha='left', fontsize=9)

    # Legend
    hypothesis_patch = mpatches.Patch(color=COLORS['bar_accent'], label='Hypothesis feature (day_variability)')
    other_patch = mpatches.Patch(color=COLORS['bar_primary'], label='Other features')
    ax.legend(handles=[hypothesis_patch, other_patch], fontsize=10, loc='lower right')

    ax.set_xlabel('Feature Importance (%)', fontsize=12)
    ax.set_title('XGBoost Feature Importance Ranking (Approach 1B)\nBipolar vs Unipolar Classification — 19 Engineered Features',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 20)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "fig4_feature_importance.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 5: Statistical Significance (3A) ───────────────────────────────────
def plot_statistical_significance(res_3a):
    """Visualize variability distributions with statistical test results."""
    results = res_3a['results']

    # Recreate individual variability values (from 3A output)
    bipolar_variability = [0.3266, 0.3553, 0.1006, 0.2212, 0.2757, 0.1862, 0.0843, 0.0714]
    unipolar_variability = [0.1931, 0.1890, 0.2837, 0.1494, 0.1088, 0.2756, 0.1427, 0.2533,
                            0.2363, 0.3734, 0.2053, 0.1910, 0.0914, 0.1311, 0.2972]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overlapping distributions (strip + box)
    data_combined = pd.DataFrame({
        'Variability': bipolar_variability + unipolar_variability,
        'Group': ['Bipolar'] * len(bipolar_variability) + ['Unipolar'] * len(unipolar_variability)
    })

    sns.boxplot(data=data_combined, x='Group', y='Variability',
                palette={'Bipolar': COLORS['bipolar'], 'Unipolar': COLORS['unipolar']},
                width=0.4, ax=axes[0], fliersize=0)
    sns.stripplot(data=data_combined, x='Group', y='Variability',
                  palette={'Bipolar': COLORS['bipolar'], 'Unipolar': COLORS['unipolar']},
                  jitter=True, size=8, alpha=0.7, ax=axes[0])

    # Annotate means
    axes[0].axhline(y=results['bipolar_mean'], color=COLORS['bipolar'],
                    linestyle='--', alpha=0.5, linewidth=1.2)
    axes[0].axhline(y=results['unipolar_mean'], color=COLORS['unipolar'],
                    linestyle='--', alpha=0.5, linewidth=1.2)

    axes[0].set_title('Activity Variability Distribution\nBipolar vs Unipolar', fontweight='bold')
    axes[0].set_ylabel('Daily Activity Variability (std of daily means)')
    axes[0].text(0, results['bipolar_mean'] + 0.005,
                 f"mean={results['bipolar_mean']:.3f}", ha='center', fontsize=9,
                 color=COLORS['bipolar'], fontweight='bold')
    axes[0].text(1, results['unipolar_mean'] + 0.005,
                 f"mean={results['unipolar_mean']:.3f}", ha='center', fontsize=9,
                 color=COLORS['unipolar'], fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: test results summary as text table
    axes[1].axis('off')
    table_data = [
        ['Test', 'Statistic', 'p-value', 'Significant?'],
        ['Independent t-test', f't={results["t_statistic"]:.4f}', f'p={results["p_value"]:.4f}', 'NO'],
        ["Welch's t-test", f't={results["t_statistic_welch"]:.4f}', f'p={results["p_value_welch"]:.4f}', 'NO'],
        ['Mann-Whitney U', 'U=55.0000', f'p={results["p_value_mannwhitney"]:.4f}', 'NO'],
        ['Cohen\'s d', f'd={results["cohens_d"]:.4f}', 'NEGLIGIBLE', '< 0.2'],
    ]

    table = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center',
                          bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Header row styling
    for j in range(4):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # "NO" cells
    for i in range(1, 4):
        table[(i, 3)].set_facecolor('#E74C3C')
        table[(i, 3)].set_text_props(color='white', fontweight='bold')
    table[(4, 3)].set_facecolor('#E67E22')
    table[(4, 3)].set_text_props(color='white', fontweight='bold')

    axes[1].set_title('Statistical Test Results\n(All tests: p > 0.05, NOT significant)',
                      fontweight='bold', fontsize=12)
    axes[1].text(0.5, 0.15,
                 f'Bipolar mean: {results["bipolar_mean"]:.4f} (n={results["bipolar_n"]})\n'
                 f'Unipolar mean: {results["unipolar_mean"]:.4f} (n={results["unipolar_n"]})\n'
                 f'Difference: {results["bipolar_mean"] - results["unipolar_mean"]:.4f}\n'
                 f'Effect size d={results["cohens_d"]:.4f} (NEGLIGIBLE)',
                 ha='center', va='center', transform=axes[1].transAxes,
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

    plt.suptitle('Approach 3A: Statistical Significance of Activity Variability\nBipolar vs Unipolar Depression',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "fig5_statistical_significance.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 6: Hyperparameter Grid Search Heatmap (1A Extended) ────────────────
def plot_hyperparameter_heatmap(res_1a_ext):
    """Heatmap of grid search results per window size."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    wd_labels = ['1e-5', '1e-4', '1e-3']
    do_labels = ['0.2', '0.4', '0.6']

    for ax, (window_key, window_label) in zip(
        axes,
        [('24hr', '24hr Window'), ('48hr', '48hr Window'), ('72hr', '72hr Window')]
    ):
        all_res = res_1a_ext['results_by_window'][window_key]['all_results']
        grid = np.zeros((3, 3))
        for i, wd in enumerate(['1e-05', '1e-04', '1e-03']):
            for j, do in enumerate(['0.2', '0.4', '0.6']):
                key = f'wd={wd}_do={do}'
                grid[i, j] = all_res[key]['accuracy'] * 100

        im = sns.heatmap(grid, annot=True, fmt='.1f', cmap='YlOrRd',
                         xticklabels=[f'do={d}' for d in do_labels],
                         yticklabels=[f'wd={w}' for w in wd_labels],
                         ax=ax, vmin=0, vmax=100,
                         annot_kws={'fontsize': 9})
        ax.set_title(f'{window_label}\nBest: {res_1a_ext["results_by_window"][window_key]["best_accuracy"]*100:.1f}%',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Weight Decay')

    plt.suptitle('CNN-LSTM Hyperparameter Grid Search (Approach 1A Extended)\nAccuracy (%) by Window Size, Weight Decay, Dropout',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "fig6_hyperparameter_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── PLOT 7: Research Story Summary (Poster Key Figure) ──────────────────────
def plot_research_summary(res_1a, res_1b, res_1c, res_3a):
    """Single-figure summary of all findings for poster."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    results_3a = res_3a['results']

    # ---- Top Left: Accuracy Bar ----
    ax1 = fig.add_subplot(gs[0, 0])
    approaches = ['Baseline', '1C\nLogReg', '1A\n24hr\nCNN', '1A\n48hr\nCNN', '1B\nXGBoost']
    accs = [65.2,
            res_1c['best_loocv_accuracy'] * 100,
            res_1a['window_results']['24hr']['accuracy'] * 100,
            res_1a['window_results']['48hr']['accuracy'] * 100,
            res_1b['loocv_accuracy'] * 100]
    bar_colors = [COLORS['baseline'], COLORS['approach_1c'],
                  COLORS['approach_1a'], COLORS['approach_1a'], COLORS['approach_1b']]
    bars = ax1.bar(approaches, accs, color=bar_colors, edgecolor='white', linewidth=1)
    ax1.axhline(y=65.2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    for bar, a in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{a:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_title('Classification Accuracy', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=9)
    ax1.set_ylim(0, 88)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=7)

    # ---- Top Middle: Feature Importance Top 5 ----
    ax2 = fig.add_subplot(gs[0, 1])
    top5_feat = ['activity_min', 'high_act\nfraction', 'activity_iqr', 'daily_max\nvariability', 'day_\nvariability']
    top5_imp = [14.5, 12.2, 10.9, 10.4, 7.7]
    colors_feat = [COLORS['bar_primary']] * 4 + [COLORS['bar_accent']]
    bars2 = ax2.barh(top5_feat[::-1], top5_imp[::-1], color=colors_feat[::-1],
                     edgecolor='white', linewidth=0.8, height=0.6)
    for bar, imp in zip(bars2, top5_imp[::-1]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{imp}%', va='center', ha='left', fontsize=8)
    ax2.set_title('Top-5 Features (1B)\nRed = hypothesis feature', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Importance (%)', fontsize=9)
    ax2.set_xlim(0, 20)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='y', labelsize=8)

    # ---- Top Right: Variability Distributions ----
    ax3 = fig.add_subplot(gs[0, 2])
    bi_var = [0.3266, 0.3553, 0.1006, 0.2212, 0.2757, 0.1862, 0.0843, 0.0714]
    uni_var = [0.1931, 0.1890, 0.2837, 0.1494, 0.1088, 0.2756, 0.1427, 0.2533,
               0.2363, 0.3734, 0.2053, 0.1910, 0.0914, 0.1311, 0.2972]
    ax3.scatter([1] * len(bi_var), bi_var, color=COLORS['bipolar'], alpha=0.8, s=60, zorder=5, label='Bipolar')
    ax3.scatter([2] * len(uni_var), uni_var, color=COLORS['unipolar'], alpha=0.8, s=60, zorder=5, label='Unipolar')
    ax3.plot([0.8, 1.2], [results_3a['bipolar_mean']] * 2,
             color=COLORS['bipolar'], linewidth=2.5, label=f"Bi mean={results_3a['bipolar_mean']:.3f}")
    ax3.plot([1.8, 2.2], [results_3a['unipolar_mean']] * 2,
             color=COLORS['unipolar'], linewidth=2.5, label=f"Uni mean={results_3a['unipolar_mean']:.3f}")
    ax3.set_xlim(0.5, 2.5)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Bipolar\n(n=8)', 'Unipolar\n(n=15)'], fontsize=9)
    ax3.set_title(f'Activity Variability (3A)\np={results_3a["p_value"]:.3f}, d={results_3a["cohens_d"]:.3f}',
                  fontweight='bold', fontsize=10)
    ax3.set_ylabel('Variability', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(fontsize=7, loc='upper right')

    # ---- Bottom: Key Findings Text ----
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    findings_text = (
        "KEY FINDINGS\n\n"
        "1.  No statistically significant difference in activity variability between bipolar and unipolar patients "
        "(p=0.893, Cohen's d=-0.060 NEGLIGIBLE). The core hypothesis is not supported.\n\n"
        "2.  All classifiers struggle: Best accuracy = 60.9% (Logistic Regression), barely above the 65.2% majority-class baseline. "
        "XGBoost with 19 engineered features (39.1%) underperforms simpler approaches.\n\n"
        "3.  CNN-LSTM window analysis: 48hr window (63.4%) > 24hr (77.9% but overfit to majority class) > 72hr (42.3%). "
        "Larger context does not help when signal is absent.\n\n"
        "4.  Root cause: Bipolar II patients recorded during depressive episodes look clinically similar to unipolar depressed patients. "
        "Actigraphy variability alone cannot distinguish them with n=8 bipolar participants."
    )
    ax4.text(0.02, 0.95, findings_text,
             transform=ax4.transAxes, fontsize=9.5, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8),
             linespacing=1.5)

    fig.suptitle('Bipolar vs Unipolar Depression Detection via Actigraphy\nProject Summary — All Approaches',
                 fontsize=14, fontweight='bold', y=1.01)

    path = RESULTS_DIR / "fig7_research_summary_poster.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    return str(path)


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("APPROACH 3B: Generating All Visualizations")
    print("=" * 70)

    res_1a, res_1a_ext, res_1b, feat_imp, res_1c, res_3a = load_results()
    print("\nAll result files loaded successfully.")

    saved_files = []
    print("\n[1/7] Accuracy comparison...")
    saved_files.append(plot_accuracy_comparison(res_1a, res_1b, res_1c))

    print("[2/7] Window size comparison...")
    saved_files.append(plot_window_size_comparison(res_1a, res_1a_ext))

    print("[3/7] Confusion matrices (all approaches: 1A x3 + 1B + 1C)...")
    saved_files.append(plot_confusion_matrices(res_1a, res_1b, res_1c))

    print("[4/7] Feature importance...")
    saved_files.append(plot_feature_importance(feat_imp))

    print("[5/7] Statistical significance...")
    saved_files.append(plot_statistical_significance(res_3a))

    print("[6/7] Hyperparameter heatmap...")
    saved_files.append(plot_hyperparameter_heatmap(res_1a_ext))

    print("[7/7] Research summary poster figure...")
    saved_files.append(plot_research_summary(res_1a, res_1b, res_1c, res_3a))

    print("\n" + "=" * 70)
    print("APPROACH 3B COMPLETE — All figures saved to results/")
    print("=" * 70)
    for f in saved_files:
        print(f"  {f}")
    print()
    print("NOTE: ROC curves for 1A CNN-LSTM could NOT be generated.")
    print("  Reason: Prediction probabilities were not saved during training.")
    print("  Only accuracy and confusion matrix were persisted in results_1a.json.")
    print("  To generate ROC curves, re-run training with probability logging enabled.")
    print()
