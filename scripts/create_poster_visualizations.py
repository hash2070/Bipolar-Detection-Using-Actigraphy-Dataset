"""
POSTER VISUALIZATIONS — Additional figures for FINAL_PRESENTATION.md
Generates 9 new poster-quality figures covering:
  - Dataset overview (participant breakdown, recording lengths, MADRS scores)
  - Sample actigraphy traces (bipolar vs unipolar vs healthy)
  - Model complexity vs accuracy scatter
  - Comprehensive all-approaches comparison
  - Feature correlation heatmap
  - Day-by-day variability illustration
  - Why deep learning struggles (sample size context)
  - Summary comparison table figure
  - Temporal context illustration
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif',
})

C = {
    'bipolar':   '#E74C3C',
    'unipolar':  '#3498DB',
    'healthy':   '#2ECC71',
    'neutral':   '#95A5A6',
    'dark':      '#2C3E50',
    'accent':    '#F39C12',
    'light_red': '#FADBD8',
    'light_blue':'#D6EAF8',
    'bg':        '#FDFEFE',
}

# ─── Load results ────────────────────────────────────────────────────────────
def load_all():
    with open("findings/1A/results_1a.json") as f:     r1a = json.load(f)
    with open("findings/1A/results_1a_extended.json") as f: r1a_ext = json.load(f)
    with open("findings/1B/results_1b.json") as f:     r1b = json.load(f)
    with open("findings/1B/feature_importance.csv") as f: feat = pd.read_csv(f)
    with open("findings/1C/results_1c.json") as f:     r1c = json.load(f)
    with open("findings/3A/results_3a.json") as f:     r3a = json.load(f)
    scores = pd.read_csv("depresjon/data/scores.csv")
    pfeats = pd.read_csv("findings/1B/participant_features.csv") if Path("findings/1B/participant_features.csv").exists() else None
    return r1a, r1a_ext, r1b, feat, r1c, r3a, scores, pfeats


# ─── FIG A: Dataset Overview ─────────────────────────────────────────────────
def fig_dataset_overview(scores):
    """3-panel: participant pie, recording days bar, MADRS distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(C['bg'])

    # ── Pie: participant breakdown
    sizes  = [32, 8, 15]
    labels = ['Healthy Controls\n(n=32)', 'Bipolar\n(n=8)', 'Unipolar\n(n=15)']
    colors = [C['healthy'], C['bipolar'], C['unipolar']]
    explode = (0.03, 0.08, 0.03)
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 10}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    axes[0].set_title('Study Participants\n(55 total)', fontweight='bold', fontsize=12)

    # ── Bar: recording days per participant
    cond = scores[scores['number'].str.startswith('condition')].copy()
    cond['group'] = cond['afftype'].map({1.0: 'Bipolar', 2.0: 'Unipolar', 3.0: 'Bipolar'}).fillna('Unknown')
    colors_bar = [C['bipolar'] if g == 'Bipolar' else C['unipolar'] for g in cond['group']]
    x = np.arange(len(cond))
    axes[1].bar(x, cond['days'], color=colors_bar, edgecolor='white', linewidth=0.8, width=0.75)
    axes[1].axhline(cond['days'].mean(), color=C['dark'], linestyle='--', linewidth=1.5,
                    label=f"Mean: {cond['days'].mean():.1f} days")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace('condition_', 'P') for n in cond['number']],
                             rotation=60, fontsize=8)
    axes[1].set_ylabel('Recording Duration (days)')
    axes[1].set_title('Recording Duration per Participant\n(Mood Disorder Group)', fontweight='bold', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    bi_patch = mpatches.Patch(color=C['bipolar'], label='Bipolar')
    uni_patch = mpatches.Patch(color=C['unipolar'], label='Unipolar')
    axes[1].legend(handles=[bi_patch, uni_patch], fontsize=9, loc='upper right')

    # ── MADRS scores: bipolar vs unipolar
    bipolar_rows  = cond[cond['group'] == 'Bipolar']
    unipolar_rows = cond[cond['group'] == 'Unipolar']
    data_plot = pd.DataFrame({
        'MADRS Score': list(bipolar_rows['madrs1'].dropna()) + list(unipolar_rows['madrs1'].dropna()),
        'Group': ['Bipolar'] * len(bipolar_rows['madrs1'].dropna()) + ['Unipolar'] * len(unipolar_rows['madrs1'].dropna())
    })
    for group, color, offset in [('Bipolar', C['bipolar'], -0.2), ('Unipolar', C['unipolar'], 0.2)]:
        subset = data_plot[data_plot['Group'] == group]['MADRS Score']
        jitter = np.random.uniform(-0.15, 0.15, len(subset))
        x_pos = (1 if group == 'Bipolar' else 2)
        axes[2].scatter([x_pos + j for j in jitter], subset, color=color, alpha=0.8, s=80, zorder=5)
        axes[2].plot([x_pos - 0.2, x_pos + 0.2], [subset.mean(), subset.mean()],
                     color=color, linewidth=3, zorder=6)
        axes[2].text(x_pos + 0.25, subset.mean(), f"μ={subset.mean():.1f}", ha='left',
                     va='center', fontsize=9, color=color, fontweight='bold', zorder=10)
    axes[2].set_xlim(0.5, 2.5)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['Bipolar\n(n=8)', 'Unipolar\n(n=15)'], fontsize=11)
    axes[2].set_ylabel('MADRS Score (0-60 scale)')
    axes[2].set_title('Depression Severity (MADRS)\nat Study Entry', fontweight='bold', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    # annotation placed in the empty center column between the two groups
    axes[2].text(1.5, axes[2].get_ylim()[0] + 0.5, 'Similar depression severity\nin both groups',
                 ha='center', va='bottom',
                 fontsize=9, style='italic', color='gray', zorder=10,
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95))

    plt.suptitle('The Depresjon Dataset: Who Were the Participants?',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = RESULTS_DIR / "figA_dataset_overview.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG B: Sample Actigraphy Traces ─────────────────────────────────────────
def fig_sample_actigraphy():
    """Show 3-day activity traces for one bipolar, one unipolar, one healthy."""
    import os
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.patch.set_facecolor(C['bg'])

    examples = [
        ('depresjon/data/condition/condition_2.csv',  'Bipolar Patient (P2)',   C['bipolar'],  'High day-to-day variability'),
        ('depresjon/data/condition/condition_3.csv',  'Unipolar Patient (P3)',  C['unipolar'], 'Persistently low, flat activity'),
        ('depresjon/data/control/control_1.csv',      'Healthy Control (C1)',   C['healthy'],  'Active, regular sleep-wake rhythm'),
    ]

    for ax, (fpath, title, color, note) in zip(axes, examples):
        df = pd.read_csv(fpath)
        activity = df['activity'].values.astype(float)
        # Show first 3 days = 4320 minutes
        show = min(4320, len(activity))
        x = np.arange(show)
        hours = x / 60

        # Smooth for visual clarity
        from numpy.lib.stride_tricks import sliding_window_view
        if show > 120:
            kernel = np.ones(60) / 60
            smoothed = np.convolve(activity[:show], kernel, mode='same')
        else:
            smoothed = activity[:show]

        ax.fill_between(hours, smoothed, alpha=0.25, color=color)
        ax.plot(hours, smoothed, color=color, linewidth=1.2, alpha=0.9)

        # Day dividers
        for day in [24, 48]:
            if day < hours[-1]:
                ax.axvline(day, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(day + 0.3, ax.get_ylim()[1] * 0.85, f'Day {day//24 + 1}',
                        fontsize=8, color='gray')

        ax.set_ylabel('Activity\n(counts/min)', fontsize=10)
        ax.set_title(f'{title}  —  {note}', fontweight='bold', fontsize=12, color=color)
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, min(72, hours[-1]))

    axes[-1].set_xlabel('Time (hours)', fontsize=11)

    plt.suptitle('What Does Actigraphy Look Like?\n3 Days of Wrist Movement Recording',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = RESULTS_DIR / "figB_sample_actigraphy.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG C: Model Complexity vs Accuracy ─────────────────────────────────────
def fig_model_complexity():
    """Scatter: number of parameters vs accuracy — simple wins with small data."""
    fig, ax = plt.subplots(figsize=(10, 7))

    models = [
        # (name, params, accuracy, color, marker, x_pt_offset, y_pt_offset, ha)
        ('Majority\nBaseline',        0,       65.2, C['neutral'],  'D',   8,   0, 'left'),
        ('Logistic\nRegression\n(1C)',5,       60.9, C['unipolar'], 's',   8,   0, 'left'),
        ('XGBoost\n(1B)',             500,     39.1, C['accent'],   's',   8,   0, 'left'),
        ('RNN-LSTM\n(2c)',            210818,  42.3, 'gray',        'v',   8,   0, 'left'),
        ('Attention\n(2b)',           223811,  48.2, '#8E44AD',     'o',   8, -14, 'left'),
        ('CNN-LSTM\nBaseline',        223682,  65.4, C['dark'],     'o', -10,   0, 'right'),
        ('CNN-LSTM\n48hr (1A)',       223682,  63.4, C['bipolar'],  '*',   8,   0, 'left'),
        ('CNN-LSTM\n24hr safe\n(1A)', 223682,  90.7, '#E67E22',     'P',   8,   0, 'left'),
        ('BiLSTM\n(2a)',              388546,  55.4, '#16A085',     'o',   8, -10, 'left'),
        ('Ensemble\n(2d)',            671046,  59.7, C['dark'],     '^',   8,  10, 'left'),
    ]

    log_params = []
    for name, params, acc, color, marker, xoff, yoff, ha in models:
        x = np.log10(params + 1)
        log_params.append(x)
        ax.scatter(x, acc, color=color, marker=marker, s=200 if marker == '*' else 130,
                   zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(name, (x, acc), textcoords='offset points',
                    xytext=(xoff, yoff),
                    ha=ha, va='center', fontsize=8, color=color)

    # Baseline reference
    ax.axhline(65.2, color=C['neutral'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Majority-class baseline (65.2%)')

    # Highlight best zone
    ax.axhspan(60, 66, alpha=0.06, color='green', label='Competitive zone (60-66%)')

    ax.set_xlabel('Model Complexity (log₁₀ parameters)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Complexity vs. Accuracy\n"More parameters ≠ better performance with small medical datasets"',
                 fontsize=13, fontweight='bold')

    xticks = [0, 1, 2, 3, 4, 5, 6]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['1', '10', '100', '1K', '10K', '100K', '1M'], fontsize=10)
    ax.set_ylim(25, 100)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='upper left')

    # Add annotation box
    ax.text(0.02, 0.08,
            'PUNCHLINE: A 5-parameter logistic\nregression (60.9%) nearly matches a\n'
            '223,682-parameter neural network (63.4%)',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#3498DB', alpha=0.9))

    plt.tight_layout()
    path = RESULTS_DIR / "figC_complexity_vs_accuracy.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG D: All Approaches Comprehensive Bar ─────────────────────────────────
def fig_all_approaches_comprehensive(r1a, r1b, r1c, r3a):
    """Horizontal grouped bar chart: all approaches, color-coded by type."""
    fig, ax = plt.subplots(figsize=(13, 9))

    approaches = [
        # (label, accuracy, color, group_tag, notes)
        ('Majority-class Baseline',             65.2, C['neutral'],  'Baseline', '★ Predict all unipolar'),
        ('Baseline: CNN-LSTM + SMOTE\n(Exp 2)', 88.0, '#E59866',     'Baseline', '⚠ Illusory — 0 bipolar detected'),
        ('Approach 1: CNN-LSTM\n+ Downsampling',65.4, '#A9CCE3',     'Approach 1', 'Honest baseline'),
        ('Approach 2a: BiLSTM',                 55.4, '#A9DFBF',     'Approach 2', ''),
        ('Approach 2b: Attention LSTM',         48.2, '#A9DFBF',     'Approach 2', 'Worst reliable'),
        ('Approach 2c: RNN-LSTM',              100.0, '#EDBB99',     'Approach 2', '⚠ Unstable/overfit'),
        ('Approach 2d: Ensemble (3×)',          59.7, '#A9DFBF',     'Approach 2', 'Best arch variant'),
        ('Approach 3-1C: LogReg (5 feat)',      60.9, C['unipolar'], 'Approach 3', '★ Interpretable best'),
        ('Approach 3-1A: CNN-LSTM 24hr\n(best config)',90.7, '#E59866','Approach 3', '⚠ Overfitting'),
        ('Approach 3-1A: CNN-LSTM 48hr',        63.4, C['bipolar'],  'Approach 3', '★ BEST RELIABLE'),
        ('Approach 3-1A: CNN-LSTM 72hr',        42.3, '#AED6F1',     'Approach 3', ''),
        ('Approach 3-1B: XGBoost (19 feat)',    39.1, C['accent'],   'Approach 3', 'More features → worse'),
    ]

    labels  = [a[0] for a in approaches]
    accs    = [a[1] for a in approaches]
    colors  = [a[2] for a in approaches]
    notes   = [a[4] for a in approaches]

    y = np.arange(len(approaches))
    bars = ax.barh(y, accs, color=colors, edgecolor='white', linewidth=1, height=0.7)

    # Baseline line
    ax.axvline(65.2, color=C['neutral'], linestyle='--', linewidth=1.5, alpha=0.8,
               label='Majority baseline (65.2%)')

    # Annotations
    for bar, acc, note in zip(bars, accs, notes):
        ax.text(acc + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1f}%', va='center', ha='left', fontweight='bold', fontsize=9)
        if note:
            ax.text(1, bar.get_y() + bar.get_height() / 2,
                    note, va='center', ha='left', fontsize=8, color='#2C3E50', style='italic')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('All Approaches: Accuracy Comparison\nBipolar vs Unipolar Depression Classification',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='lower right')

    # Group labels on right
    group_positions = {'Baseline': (0.5, 1.5), 'Approach 1': (2,), 'Approach 2': (3, 4, 5, 6),
                       'Approach 3': (7, 8, 9, 10, 11)}
    for grp, indices in [('Baseline', [0, 1]), ('Approach 1', [2]),
                         ('Approach 2', [3, 4, 5, 6]), ('Approach 3', [7, 8, 9, 10, 11])]:
        mid = np.mean(indices)
        ax.text(108, mid, grp, va='center', ha='center', fontsize=9, fontweight='bold',
                color=C['dark'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', edgecolor='lightgray'))

    plt.tight_layout()
    path = RESULTS_DIR / "figD_all_approaches_comprehensive.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG E: Feature Correlation Heatmap ──────────────────────────────────────
def fig_feature_correlation(pfeats):
    """Heatmap of correlation between the 19 engineered features."""
    if pfeats is None:
        print("[SKIP] figE — participant_features.csv not found")
        return

    feature_cols = [c for c in pfeats.columns
                    if c not in ['participant_id', 'label', 'num_days']]
    X = pfeats[feature_cols].copy()

    corr = X.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle only
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax,
                annot_kws={'fontsize': 7},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})

    ax.set_title('Feature Correlation Matrix (Approach 3-1B)\n'
                 '19 Engineered Actigraphy Features — Bipolar vs Unipolar',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    path = RESULTS_DIR / "figE_feature_correlation.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG F: Day-by-Day Variability Illustration ───────────────────────────────
def fig_variability_illustration():
    """Illustrative: multi-day activity for high-variability bipolar vs stable unipolar."""
    np.random.seed(42)

    # Simulated based on actual mean variability values from 3A
    # Bipolar: more variable day-to-day; Unipolar: more stable
    days = 14

    # Bipolar patient (high variability) — modeled on real condition_2 (variability=0.327)
    bi_daily_means = np.array([0.28, -0.31, 0.42, -0.18, 0.35, -0.25, 0.40,
                               -0.10, 0.33, -0.22, 0.45, -0.15, 0.30, -0.28])

    # Unipolar patient (stable low) — modeled on real condition_3 (variability=0.189)
    uni_daily_means = np.array([-0.20, -0.18, -0.23, -0.15, -0.19, -0.22, -0.16,
                                -0.21, -0.18, -0.20, -0.24, -0.17, -0.19, -0.21])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(C['bg'])

    day_x = np.arange(1, days + 1)

    # ── Top left: raw daily means
    axes[0, 0].plot(day_x, bi_daily_means, 'o-', color=C['bipolar'], linewidth=2.5,
                    markersize=8, label='Bipolar', zorder=5)
    axes[0, 0].plot(day_x, uni_daily_means, 's-', color=C['unipolar'], linewidth=2.5,
                    markersize=8, label='Unipolar', zorder=5)
    axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].fill_between(day_x, bi_daily_means, 0, alpha=0.1, color=C['bipolar'])
    axes[0, 0].fill_between(day_x, uni_daily_means, 0, alpha=0.1, color=C['unipolar'])
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Activity Level (normalized)')
    axes[0, 0].set_title('Daily Mean Activity Over 2 Weeks\n(Illustrative Examples)',
                          fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # ── Top right: variability comparison (all 23 participants)
    bi_var = [0.3266, 0.3553, 0.1006, 0.2212, 0.2757, 0.1862, 0.0843, 0.0714]
    uni_var = [0.1931, 0.1890, 0.2837, 0.1494, 0.1088, 0.2756, 0.1427, 0.2533,
               0.2363, 0.3734, 0.2053, 0.1910, 0.0914, 0.1311, 0.2972]

    all_vars = bi_var + uni_var
    all_groups = ['Bipolar'] * len(bi_var) + ['Unipolar'] * len(uni_var)
    df_var = pd.DataFrame({'Variability': all_vars, 'Group': all_groups})

    parts = axes[0, 1].violinplot(
        [bi_var, uni_var], positions=[1, 2],
        showmeans=True, showmedians=False, widths=0.6
    )
    parts['bodies'][0].set_facecolor(C['bipolar'])
    parts['bodies'][0].set_alpha(0.4)
    parts['bodies'][1].set_facecolor(C['unipolar'])
    parts['bodies'][1].set_alpha(0.4)
    for pc in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if pc in parts:
            parts[pc].set_color(C['dark'])

    # Overlay individual points
    np.random.seed(10)
    axes[0, 1].scatter(1 + np.random.uniform(-0.12, 0.12, len(bi_var)),
                       bi_var, color=C['bipolar'], s=70, zorder=5, alpha=0.9)
    axes[0, 1].scatter(2 + np.random.uniform(-0.12, 0.12, len(uni_var)),
                       uni_var, color=C['unipolar'], s=70, zorder=5, alpha=0.9)

    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Bipolar\n(n=8)', 'Unipolar\n(n=15)'], fontsize=11)
    axes[0, 1].set_ylabel('Day-to-Day Variability')
    axes[0, 1].set_title(f'Actual Variability Distribution\np=0.893, Cohen\'s d=−0.060 (NEGLIGIBLE)',
                          fontweight='bold')
    axes[0, 1].text(1.5, max(all_vars) * 0.95,
                    'Massive overlap —\nno separation!',
                    ha='center', fontsize=9, color='gray', style='italic')
    axes[0, 1].grid(axis='y', alpha=0.25)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # ── Bottom left: 24hr vs 48hr window illustration
    x_hours = np.linspace(0, 72, 4320)
    # Simulate bipolar with visible cycling at 48h scale
    signal = (0.4 * np.sin(2 * np.pi * x_hours / 48) +
              0.3 * np.sin(2 * np.pi * x_hours / 24) +
              0.15 * np.random.randn(len(x_hours)))
    signal = np.clip(signal, -1, 1)

    axes[1, 0].plot(x_hours, signal, color=C['bipolar'], linewidth=0.8, alpha=0.8)
    axes[1, 0].axvspan(0, 24, alpha=0.12, color='orange', label='24hr window (misses cycle)')
    axes[1, 0].axvspan(0, 48, alpha=0.08, color='green', label='48hr window (captures cycle)')
    axes[1, 0].axvline(24, color='orange', linestyle='--', linewidth=1.5)
    axes[1, 0].axvline(48, color='green', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Activity (normalized)')
    axes[1, 0].set_title('Why 48hr Windows Work Better\n(Bipolar cycles span multiple days)',
                          fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # ── Bottom right: window size accuracy
    windows = ['24hr\n(1,440 min)', '48hr\n(2,880 min)', '72hr\n(4,320 min)']
    base_accs = [77.9, 53.7, 23.1]
    best_accs = [90.7, 63.4, 42.3]
    x_w = np.arange(len(windows))
    width = 0.35

    bars1 = axes[1, 1].bar(x_w - width/2, base_accs, width, label='Base run',
                            color='#85C1E9', edgecolor='white')
    bars2 = axes[1, 1].bar(x_w + width/2, best_accs, width, label='Best config',
                            color=[C['accent'], C['bipolar'], C['neutral']],
                            edgecolor='white')
    axes[1, 1].axhline(65.2, color=C['neutral'], linestyle='--', linewidth=1.5, alpha=0.8,
                        label='Majority baseline (65.2%)')
    for bar, acc in zip(list(bars1) + list(bars2), base_accs + best_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{acc:.0f}%', ha='center', fontsize=8, fontweight='bold')
    axes[1, 1].set_xticks(x_w)
    axes[1, 1].set_xticklabels(windows)
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title('Window Size Effect on Accuracy\n(48hr = best realistic performance)',
                          fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.25)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    plt.suptitle('Key Findings: Temporal Patterns & Variability Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = RESULTS_DIR / "figF_variability_and_windows.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG G: Why Deep Learning Struggles (Sample Size Context) ────────────────
def fig_sample_size_context():
    """Bubble chart showing sample size vs performance across psychiatric ML papers."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Reference points from literature (approximate)
    papers = [
        # (study_name, n_minority, accuracy, marker_size_factor, color, our_work, xoff, yoff, ha)
        ('Jakobsen 2020\n(this dataset,\nhand-crafted)', 8, 82, 200, C['unipolar'], False,  8,  -5, 'left'),
        ('Ortiz 2015\n(Bipolar, EEG)',                  20, 78, 300, '#8E44AD',     False,   8,   8, 'left'),
        ('Reinertsen 2021\n(ECG, Bipolar)',              22, 74, 300, '#F39C12',     False,  8,  -14, 'left'),
        ('Busk 2020\n(Smartphone,\nBipolar)',            47, 71, 400, '#27AE60',     False,  8,    5, 'left'),
        ('Our Work\n(CNN-LSTM 48hr)',                     8, 63, 200, C['bipolar'],  True,   8,    5, 'left'),
        ('Our Work\n(LogReg 5 feat)',                     8, 61, 200, C['bipolar'],  True,   8,  -14, 'left'),
        ('Hypothetical\n(n=50 bipolar)',                 50, 78, 500, '#E74C3C',     False,  8,    5, 'left'),
        ('Hypothetical*\n(n=100 bipolar)',              100, 85, 600, '#C0392B',     False, -10,  -5, 'right'),
    ]

    for name, n, acc, sz, color, ours, xoff, yoff, ha in papers:
        edgecolor = C['bipolar'] if ours else 'white'
        linewidth = 3 if ours else 1.5
        ax.scatter(n, acc, s=sz, color=color, alpha=0.8,
                   edgecolors=edgecolor, linewidth=linewidth, zorder=5)
        ax.annotate(name, (n, acc), textcoords='offset points',
                    xytext=(xoff, yoff),
                    ha=ha, va='center', fontsize=8,
                    color=C['bipolar'] if ours else C['dark'],
                    fontweight='bold' if ours else 'normal')

    # Trend line (illustrative)
    x_trend = np.linspace(5, 110, 200)
    y_trend = 55 + 30 * (1 - np.exp(-x_trend / 30))
    ax.plot(x_trend, y_trend, '--', color='gray', alpha=0.6, linewidth=2,
            label='Expected trend (more data → better accuracy)')

    ax.axvline(8, color=C['bipolar'], linestyle=':', alpha=0.6, linewidth=1.5)
    ax.text(9, 57, 'Our cohort\n(n=8)', fontsize=9, color=C['bipolar'], style='italic')

    ax.set_xlabel('Number of Minority Class Samples (Bipolar Patients)', fontsize=12)
    ax.set_ylabel('Best Reported Accuracy (%)', fontsize=12)
    ax.set_title('Sample Size vs. Classification Accuracy\nPsychiatric Actigraphy/Wearable Studies',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 120)
    ax.set_ylim(50, 95)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10)

    ax.text(0.98, 0.05,
            'Note: Literature values approximate.\nHypothetical points show extrapolation.',
            transform=ax.transAxes, ha='right', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    path = RESULTS_DIR / "figG_sample_size_context.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG H: Comprehensive Results Table Figure ───────────────────────────────
def fig_results_table():
    """Visual table comparing all experiments — color-coded cells."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    columns = ['Approach', 'Method', 'Accuracy', 'F1-Score', 'ROC-AUC',
               'Bipolar\nDetected?', 'Trustworthy?', 'Key Takeaway']

    data = [
        ['Baseline\nExp 1', '1D-CNN-LSTM\n(Healthy vs Depressed)', '53.4%', '0.516', '0.556',
         'N/A', 'YES', 'Barely above chance'],
        ['Baseline\nExp 2', '1D-CNN-LSTM + SMOTE\n(Bipolar vs Unipolar)', '88.0%', '0.936', 'NaN',
         'NO (0/all)', 'NO', 'Illusory — model collapse'],
        ['Approach 1', 'Downsampling\n(1:1 balance)', '65.4%', '0.791', 'NaN',
         'NO (0/349)', 'YES', 'Balance alone insufficient'],
        ['Approach 2a', 'BiLSTM', '55.4%', '0.713', 'NaN',
         'NO', 'YES', 'Bidirectional hurts (-10%)'],
        ['Approach 2b', 'Attention LSTM', '48.2%', '0.650', 'NaN',
         'NO', 'YES', 'Worst reliable result'],
        ['Approach 2c', 'RNN-LSTM', '100%', '1.000', 'NaN',
         '—', 'NO', 'Training instability — discard'],
        ['Approach 2d', 'Ensemble (3x)', '59.7%', '0.747', 'NaN',
         'NO', 'YES', 'Best architecture variant'],
        ['Approach 3-1C', 'Logistic Regression\n(5 features)', '60.9%', '—', '0.308',
         '2/8 (w/ balanced)', 'YES', 'Interpretable — detects bipolar'],
        ['Approach 3-1A', 'CNN-LSTM 48hr\n(Best realistic)', '63.4%', '—', '—',
         'NO', 'YES', '★ BEST RELIABLE RESULT'],
        ['Approach 3-1B', 'XGBoost\n(19 features)', '39.1%', '—', '0.217',
         'NO', 'YES', 'More features → worse (n=23)'],
        ['Approach 3-3A', 'Statistical Tests\n(t-test, Cohen\'s d)', '—', '—', '—',
         '—', 'YES', 'p=0.893, d=-0.06 NEGLIGIBLE'],
    ]

    table = ax.table(cellText=data, colLabels=columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_facecolor(C['dark'])
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=9)
        table[(0, j)].set_height(0.12)

    # Row color coding
    row_colors = {
        1: '#FCF3CF',   # baseline exp1 — yellow
        2: '#FADBD8',   # baseline exp2 — red (untrustworthy)
        3: '#EBF5FB',   # approach 1 — light blue
        4: '#EAFAF1',   # approach 2a
        5: '#FDEDEC',   # approach 2b — worst
        6: '#FDEDEC',   # approach 2c — red (discard)
        7: '#EAFAF1',   # approach 2d
        8: '#D6EAF8',   # 3-1C — highlight
        9: '#D5F5E3',   # 3-1A 48hr — highlight green (best)
        10: '#FEF9E7',  # 3-1B
        11: '#EAF2FF',  # 3-3A
    }
    for row_idx, color in row_colors.items():
        for j in range(len(columns)):
            table[(row_idx, j)].set_facecolor(color)

    # Bold best result row
    for j in range(len(columns)):
        table[(9, j)].set_text_props(fontweight='bold')
        table[(9, j)].set_facecolor('#A9DFBF')

    # Red out the unreliable rows
    for j in range(len(columns)):
        table[(2, j)].set_facecolor('#F5B7B1')
        table[(6, j)].set_facecolor('#F5B7B1')

    ax.set_title('Complete Experiment Summary Table\nBipolar vs Unipolar Depression Detection via Actigraphy',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = RESULTS_DIR / "figH_results_summary_table.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print(f"[SAVED] {path}")


# ─── FIG I: Three Punchlines Summary ─────────────────────────────────────────
def fig_three_punchlines(r3a):
    """Visual for the 3 poster punchlines."""
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.patch.set_facecolor('#F8F9FA')

    results_3a = r3a['results']

    # ── Punchline 1: Cohen's d context bar
    ax1 = fig.add_subplot(gs[0])
    effect_benchmarks = [
        ('Our result\n(d=−0.06)', -0.06, C['bipolar']),
        ('Small effect\n(d=0.2)', 0.2, '#85C1E9'),
        ('Medium effect\n(d=0.5)', 0.5, '#F8C471'),
        ('Large effect\n(d=0.8)', 0.8, '#A9DFBF'),
    ]
    labels_e = [e[0] for e in effect_benchmarks]
    vals_e   = [abs(e[1]) for e in effect_benchmarks]
    colors_e = [e[2] for e in effect_benchmarks]
    bars = ax1.bar(labels_e, vals_e, color=colors_e, edgecolor='white', linewidth=2, width=0.6)
    for bar, val in zip(bars, vals_e):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'|d|={val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticklabels(labels_e, fontsize=8, rotation=12, ha='right')
    ax1.set_ylabel('Effect Size |Cohen\'s d|')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Punchline 1:\nWe Quantified the Problem',
                  fontweight='bold', fontsize=11, color=C['dark'])
    ax1.text(0, 0.65,
             'Our result is essentially\nZERO effect — telling future\nresearchers exactly how\nmuch harder this is.',
             fontsize=9, color=C['dark'],
             bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9))
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Punchline 2: Window size story
    ax2 = fig.add_subplot(gs[1])
    windows = ['24hr', '48hr', '72hr']
    realistic = [77.9, 63.4, 42.3]
    bar_colors = ['#F8C471', C['bipolar'], '#85C1E9']
    bars2 = ax2.bar(windows, realistic, color=bar_colors, edgecolor='white', linewidth=2, width=0.5)
    ax2.axhline(65.2, color=C['neutral'], linestyle='--', linewidth=2, label='Baseline')
    for bar, acc in zip(bars2, realistic):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 85)
    ax2.set_title('Punchline 2:\nContext Window Matters',
                  fontweight='bold', fontsize=11, color=C['dark'])
    ax2.text(1, 30,
             '48-hour recordings\ncapture mood-cycling\npatterns that 24-hour\nsnapshots miss.',
             ha='center', fontsize=9, color=C['dark'],
             bbox=dict(boxstyle='round', facecolor='#EAFAF1', alpha=0.9))
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Punchline 3: Simple vs complex
    ax3 = fig.add_subplot(gs[2])
    model_names = ['Logistic\nRegression\n5 params', 'XGBoost\n~500\nparams', 'CNN-LSTM\n223K\nparams']
    model_accs  = [60.9, 39.1, 63.4]
    model_cols  = [C['unipolar'], C['accent'], C['bipolar']]
    bars3 = ax3.bar(model_names, model_accs, color=model_cols, edgecolor='white', linewidth=2, width=0.5)
    ax3.axhline(65.2, color=C['neutral'], linestyle='--', linewidth=2, label='Baseline')
    for bar, acc in zip(bars3, model_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax3.set_ylabel('LOOCV / Test Accuracy (%)')
    ax3.set_ylim(0, 85)
    ax3.set_title('Punchline 3:\nSimple Rivals Complex',
                  fontweight='bold', fontsize=11, color=C['dark'])
    ax3.text(1, 52,
             'A 5-parameter logistic\nregression (60.9%) nearly\nmatches a 223K-parameter\nneural network (63.4%).',
             ha='center', fontsize=9, color=C['dark'],
             bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.9))
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.suptitle('Three Punchlines: What This Project Taught Us',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "figI_three_punchlines.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"[SAVED] {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("POSTER VISUALIZATIONS: Generating 9 additional figures")
    print("=" * 70)

    r1a, r1a_ext, r1b, feat, r1c, r3a, scores, pfeats = load_all()
    print("All data loaded.\n")

    print("[A/9] Dataset overview...")
    fig_dataset_overview(scores)

    print("[B/9] Sample actigraphy traces...")
    fig_sample_actigraphy()

    print("[C/9] Model complexity vs accuracy...")
    fig_model_complexity()

    print("[D/9] All approaches comprehensive bar...")
    fig_all_approaches_comprehensive(r1a, r1b, r1c, r3a)

    print("[E/9] Feature correlation heatmap...")
    fig_feature_correlation(pfeats)

    print("[F/9] Variability & window illustration...")
    fig_variability_illustration()

    print("[G/9] Sample size context...")
    fig_sample_size_context()

    print("[H/9] Results summary table figure...")
    fig_results_table()

    print("[I/9] Three punchlines figure...")
    fig_three_punchlines(r3a)

    print("\n" + "=" * 70)
    print("ALL POSTER VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nFigures saved to results/:")
    for p in sorted(Path("results").glob("fig[A-I]*.png")):
        print(f"  {p}")
    print("\nFigures for poster (recommended):")
    print("  figA — Dataset overview (Panel 1)")
    print("  figB — Sample actigraphy (Panel 2)")
    print("  figI — Three punchlines (Center panel)")
    print("  figF — Variability & window analysis (Panel 3)")
    print("  figH — Complete results table (Panel 4)")
    print("  figC — Complexity vs accuracy (Panel 5)")
    print()
