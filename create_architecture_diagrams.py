"""
ARCHITECTURE DIAGRAMS — All models used in the project.

Generates:
  figJ_arch_baseline_cnnlstm.png   — CNN-LSTM Baseline (individual, detailed)
  figK_arch_bilstm.png             — BiLSTM (individual, detailed)
  figL_arch_attention.png          — Attention LSTM (individual, detailed)
  figM_arch_rnnlstm.png            — RNN-LSTM (individual, detailed)
  figN_arch_ensemble.png           — Ensemble CNN-LSTM (individual)
  figO_arch_classical_ml.png       — XGBoost + Logistic Regression pipelines
  figP_arch_all_comparison.png     — All 5 neural architectures side-by-side
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
CLR = {
    'input':    ('#D5E8D4', '#82B366'),   # face, edge
    'conv':     ('#DAE8FC', '#6C8EBF'),
    'bn_relu':  ('#E8DEF8', '#7B5EA7'),
    'pool':     ('#FFE6CC', '#D6B656'),
    'lstm':     ('#FFF2CC', '#D6B656'),
    'bilstm':   ('#F8CECC', '#B85450'),
    'rnn':      ('#D5E8D4', '#82B366'),
    'attn':     ('#E1D5E7', '#9673A6'),
    'fc':       ('#DAE8FC', '#6C8EBF'),
    'dropout':  ('#F5F5F5', '#666666'),
    'output':   ('#F8CECC', '#B85450'),
    'arrow':    '#555555',
    'bg':       '#FAFAFA',
    'dim':      '#555555',
}

# ── Drawing helpers ────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel='', color_key='fc',
        fontsize=9, bold=False):
    """Draw a rounded-corner box at (x, y) with width w, height h."""
    face, edge = CLR[color_key]
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.03",
                           facecolor=face, edgecolor=edge,
                           linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y + (0.04 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fw, color='#222222', zorder=4)
    if sublabel:
        ax.text(x, y - 0.11, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#555555', zorder=4,
                style='italic')


def arrow(ax, x1, y1, x2, y2, label=''):
    """Draw a vertical arrow from (x1,y1) down to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=CLR['arrow'],
                                lw=1.6, connectionstyle='arc3,rad=0.0'),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.18, my, label, ha='left', va='center',
                fontsize=7.5, color=CLR['dim'], zorder=5)


def section_label(ax, x, y, text, color='#2C3E50'):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=8, color=color, fontweight='bold',
            bbox=dict(facecolor='#ECF0F1', edgecolor='lightgray',
                      boxstyle='round,pad=0.25', linewidth=0.8))


# ── FIGURE J: Baseline CNN-LSTM ────────────────────────────────────────────
def fig_cnnlstm():
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.set_xlim(0, 6); ax.set_ylim(-0.5, 13.5)
    ax.axis('off')
    fig.patch.set_facecolor(CLR['bg'])

    cx = 3.0          # centre x
    W, H = 4.2, 0.55  # box width, height
    gap = 0.95         # step between box centres

    layers = [
        # (y,  label,               sublabel,                    ckey,     bold)
        (13.0, 'INPUT',             '[batch × 1 × 1,440 min]',  'input',  True),
        (12.0, 'Conv1D  (k=7, 64)', 'stride=1, padding=3',      'conv',   False),
        (11.2, 'BatchNorm + ReLU',  '',                          'bn_relu',False),
        (10.5, 'MaxPool1D (k=2)',   '[batch × 64 × 720]',        'pool',   False),
        ( 9.5, 'Conv1D  (k=5, 128)','stride=1, padding=2',       'conv',   False),
        ( 8.7, 'BatchNorm + ReLU',  '',                          'bn_relu',False),
        ( 8.0, 'MaxPool1D (k=2)',   '[batch × 128 × 360]',       'pool',   False),
        ( 7.0, 'Transpose',         '[batch × 360 × 128]',       'dropout',False),
        ( 6.0, 'LSTM',              'hidden=128  →  final h',    'lstm',   False),
        ( 5.0, 'Flatten',           '[batch × 128]',             'dropout',False),
        ( 4.0, 'FC  128 → 256',     'ReLU  +  Dropout(0.4)',     'fc',     False),
        ( 3.0, 'FC  256 → 64',      'ReLU  +  Dropout(0.4)',     'fc',     False),
        ( 2.0, 'Linear  64 → 2',    '',                          'output', False),
        ( 1.0, 'Softmax',           '',                          'output', False),
        ( 0.0, 'OUTPUT',            'Bipolar  /  Unipolar',      'output', True),
    ]

    prev_y = None
    for (y, lbl, sub, ck, bd) in layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=9, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    # Section brackets
    for (y0, y1, txt, col) in [
        (10.9, 12.3,  'CNN BLOCK 1', CLR['conv'][1]),
        ( 8.2, 10.2,  'CNN BLOCK 2', CLR['conv'][1]),
        ( 5.5,  7.3,  'LSTM',        CLR['lstm'][1]),
        ( 0.5,  4.5,  'CLASSIFIER',  CLR['output'][1]),
    ]:
        ax.plot([5.6, 5.6], [y0, y1], color=col, lw=2, solid_capstyle='round')
        ax.text(5.75, (y0+y1)/2, txt, va='center', fontsize=8,
                color=col, fontweight='bold', rotation=90)

    ax.set_title('Baseline: CNN-LSTM Classifier\n223,682 parameters',
                 fontsize=13, fontweight='bold', pad=10)

    # Param counts aside
    for txt, y in [('64 feature maps', 11.6), ('128 feature maps', 9.1),
                   ('128 hidden units', 6.0)]:
        ax.text(0.3, y, txt, ha='left', va='center', fontsize=7.5,
                color=CLR['dim'], style='italic')

    plt.tight_layout()
    p = RESULTS_DIR / "figJ_arch_baseline_cnnlstm.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE K: BiLSTM ───────────────────────────────────────────────────────
def fig_bilstm():
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.set_xlim(0, 6); ax.set_ylim(-0.5, 13.5)
    ax.axis('off')
    fig.patch.set_facecolor(CLR['bg'])

    cx, W, H = 3.0, 4.2, 0.55

    layers = [
        (13.0, 'INPUT',               '[batch × 1 × 1,440 min]',  'input',  True),
        (12.0, 'Conv1D  (k=7, 64)',   'stride=1, padding=3',       'conv',   False),
        (11.2, 'BatchNorm + ReLU',    '',                          'bn_relu',False),
        (10.5, 'MaxPool1D (k=2)',     '[batch × 64 × 720]',        'pool',   False),
        ( 9.5, 'Conv1D  (k=5, 128)',  'stride=1, padding=2',       'conv',   False),
        ( 8.7, 'BatchNorm + ReLU',    '',                          'bn_relu',False),
        ( 8.0, 'MaxPool1D (k=2)',     '[batch × 128 × 360]',       'pool',   False),
        ( 7.0, 'Transpose',           '[batch × 360 × 128]',       'dropout',False),
        ( 6.0, 'BiLSTM',              'hidden=128 fwd + 128 bwd',  'bilstm', True),
        ( 5.0, 'Concat fwd + bwd',    '[batch × 256]',             'bilstm', False),
        ( 4.0, 'FC  256 → 256',       'ReLU  +  Dropout(0.4)',     'fc',     False),
        ( 3.0, 'FC  256 → 64',        'ReLU  +  Dropout(0.4)',     'fc',     False),
        ( 2.0, 'Linear  64 → 2',      '',                          'output', False),
        ( 1.0, 'Softmax',             '',                          'output', False),
        ( 0.0, 'OUTPUT',              'Bipolar  /  Unipolar',      'output', True),
    ]

    prev_y = None
    for (y, lbl, sub, ck, bd) in layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=9, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    # BiLSTM split arrows
    for (y0, y1, txt, col) in [
        (10.9, 12.3,  'CNN BLOCK 1', CLR['conv'][1]),
        ( 8.2, 10.2,  'CNN BLOCK 2', CLR['conv'][1]),
        ( 4.6,  7.3,  'BI-LSTM',     CLR['bilstm'][1]),
        ( 0.5,  4.1,  'CLASSIFIER',  CLR['output'][1]),
    ]:
        ax.plot([5.6, 5.6], [y0, y1], color=col, lw=2, solid_capstyle='round')
        ax.text(5.75, (y0+y1)/2, txt, va='center', fontsize=8,
                color=col, fontweight='bold', rotation=90)

    # Annotate the two directions inside BiLSTM box
    ax.text(2.0, 6.0, '→ forward', ha='center', va='center',
            fontsize=7, color=CLR['bilstm'][1], style='italic')
    ax.text(4.0, 6.0, '← backward', ha='center', va='center',
            fontsize=7, color=CLR['bilstm'][1], style='italic')

    ax.set_title('Architecture 2a: Bidirectional LSTM\n388,546 parameters  (+74% vs baseline)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    p = RESULTS_DIR / "figK_arch_bilstm.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE L: Attention LSTM ───────────────────────────────────────────────
def fig_attention():
    fig, ax = plt.subplots(figsize=(7, 15))
    ax.set_xlim(0, 7); ax.set_ylim(-0.5, 14.5)
    ax.axis('off')
    fig.patch.set_facecolor(CLR['bg'])

    cx, W, H = 3.5, 4.6, 0.55

    layers = [
        (14.0, 'INPUT',                 '[batch × 1 × 1,440 min]',          'input',  True),
        (13.0, 'Conv1D  (k=7, 64)',     'stride=1, padding=3',               'conv',   False),
        (12.2, 'BatchNorm + ReLU',      '',                                  'bn_relu',False),
        (11.5, 'MaxPool1D (k=2)',       '[batch × 64 × 720]',                'pool',   False),
        (10.5, 'Conv1D  (k=5, 128)',    'stride=1, padding=2',               'conv',   False),
        ( 9.7, 'BatchNorm + ReLU',      '',                                  'bn_relu',False),
        ( 9.0, 'MaxPool1D (k=2)',       '[batch × 128 × 360]',               'pool',   False),
        ( 8.0, 'Transpose',             '[batch × 360 × 128]',               'dropout',False),
        ( 7.0, 'LSTM',                  'hidden=128  →  ALL outputs',        'lstm',   False),
        ( 5.8, 'Attention Layer',       'Linear(128→1) + Softmax → weights', 'attn',   True),
        ( 4.8, 'Weighted Sum',          '[batch × 128]  (context vector)',   'attn',   False),
        ( 3.8, 'FC  128 → 256',         'ReLU  +  Dropout(0.4)',             'fc',     False),
        ( 2.8, 'FC  256 → 64',          'ReLU  +  Dropout(0.4)',             'fc',     False),
        ( 1.8, 'Linear  64 → 2',        '',                                  'output', False),
        ( 0.8, 'Softmax',               '',                                  'output', False),
        ( 0.0, 'OUTPUT',                'Bipolar  /  Unipolar',              'output', True),
    ]

    prev_y = None
    for (y, lbl, sub, ck, bd) in layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=9, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    # Attention loop arrow (LSTM output feeds Attention)
    ax.annotate('', xy=(6.6, 5.8), xytext=(6.6, 7.0),
                arrowprops=dict(arrowstyle='->', color=CLR['attn'][1],
                                lw=1.6, linestyle='dashed'))
    ax.text(6.7, 6.4, 'LSTM\noutputs\n(all\nsteps)', ha='left', va='center',
            fontsize=7, color=CLR['attn'][1], style='italic')

    for (y0, y1, txt, col) in [
        (11.9, 13.3, 'CNN BLOCK 1', CLR['conv'][1]),
        ( 9.2, 11.2, 'CNN BLOCK 2', CLR['conv'][1]),
        ( 4.4,  8.3, 'LSTM+ATTN',  CLR['attn'][1]),
        ( 0.4,  4.0, 'CLASSIFIER', CLR['output'][1]),
    ]:
        ax.plot([6.3, 6.3], [y0, y1], color=col, lw=2, solid_capstyle='round')
        ax.text(6.45, (y0+y1)/2, txt, va='center', fontsize=8,
                color=col, fontweight='bold', rotation=90)

    ax.set_title('Architecture 2b: Attention LSTM\n223,811 parameters  (+0.06% vs baseline)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    p = RESULTS_DIR / "figL_arch_attention.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE M: RNN-LSTM ─────────────────────────────────────────────────────
def fig_rnnlstm():
    fig, ax = plt.subplots(figsize=(6, 13))
    ax.set_xlim(0, 6); ax.set_ylim(-0.5, 12.5)
    ax.axis('off')
    fig.patch.set_facecolor(CLR['bg'])

    cx, W, H = 3.0, 4.2, 0.55

    layers = [
        (12.0, 'INPUT',              '[batch × 1 × 1,440 min]',    'input',  True),
        (11.0, 'Transpose',          '[batch × 1,440 × 1]',        'dropout',False),
        (10.0, 'SimpleRNN  (1 → 64)','ReLU  activation',           'rnn',    False),
        ( 9.0, 'RNN output',         '[batch × 1,440 × 64]',       'rnn',    False),
        ( 8.0, 'SimpleRNN (64 → 128)','ReLU  activation',          'rnn',    False),
        ( 7.0, 'RNN output',         '[batch × 1,440 × 128]',      'rnn',    False),
        ( 6.0, 'LSTM',               'hidden=128  →  final h',     'lstm',   False),
        ( 5.0, 'Flatten',            '[batch × 128]',              'dropout',False),
        ( 4.0, 'FC  128 → 256',      'ReLU  +  Dropout(0.4)',      'fc',     False),
        ( 3.0, 'FC  256 → 64',       'ReLU  +  Dropout(0.4)',      'fc',     False),
        ( 2.0, 'Linear  64 → 2',     '',                           'output', False),
        ( 1.0, 'Softmax',            '',                           'output', False),
        ( 0.0, 'OUTPUT',             'Bipolar  /  Unipolar',       'output', True),
    ]

    prev_y = None
    for (y, lbl, sub, ck, bd) in layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=9, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    for (y0, y1, txt, col) in [
        ( 8.6, 11.3, 'RNN STACK',   CLR['rnn'][1]),
        ( 5.6,  8.3, 'LSTM',        CLR['lstm'][1]),
        ( 0.5,  4.5, 'CLASSIFIER',  CLR['output'][1]),
    ]:
        ax.plot([5.6, 5.6], [y0, y1], color=col, lw=2, solid_capstyle='round')
        ax.text(5.75, (y0+y1)/2, txt, va='center', fontsize=8,
                color=col, fontweight='bold', rotation=90)

    # NO CNN badge
    ax.text(0.4, 9.5, 'No CNN\nin this\narchitecture', ha='center', va='center',
            fontsize=8, color='#C0392B', fontweight='bold',
            bbox=dict(facecolor='#FADBD8', edgecolor='#E74C3C',
                      boxstyle='round,pad=0.3', linewidth=1.2))

    # Vanishing gradient warning
    ax.text(0.4, 7.0, '⚠ Vanishing\ngradient\nrisk', ha='center', va='center',
            fontsize=7.5, color='#E67E22',
            bbox=dict(facecolor='#FEF9E7', edgecolor='#F39C12',
                      boxstyle='round,pad=0.3', linewidth=1.0))

    ax.set_title('Architecture 2c: RNN-LSTM\n210,818 parameters  (−5.7% vs baseline)\n⚠ Training unstable — result discarded',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    p = RESULTS_DIR / "figM_arch_rnnlstm.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE N: Ensemble ─────────────────────────────────────────────────────
def fig_ensemble():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    fig.patch.set_facecolor(CLR['bg'])

    H = 0.55
    seeds = [42, 43, 44]
    xs = [2.3, 7.0, 11.7]

    # Draw 3 CNN-LSTM towers
    mini_layers = [
        ('INPUT\n[B×1×1440]', 'input'),
        ('Conv1D(k=7,64)\n+BN+ReLU+Pool', 'conv'),
        ('Conv1D(k=5,128)\n+BN+ReLU+Pool', 'conv'),
        ('LSTM(128)', 'lstm'),
        ('FC 128→256→64', 'fc'),
        ('Linear 64→2', 'output'),
        ('Logits', 'output'),
    ]

    ys = [7.5, 6.4, 5.3, 4.2, 3.1, 2.0, 1.0]

    for xi, seed in zip(xs, seeds):
        ax.text(xi, 8.2, f'Model {seeds.index(seed)+1}\n(seed={seed})',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='#2C3E50',
                bbox=dict(facecolor='#EBF5FB', edgecolor='#3498DB',
                          boxstyle='round,pad=0.3', linewidth=1.5))
        prev_y = None
        for (lbl, ck), y in zip(mini_layers, ys):
            box(ax, xi, y, 3.2, 0.65, lbl, '', ck, fontsize=8)
            if prev_y is not None:
                arrow(ax, xi, prev_y - H/2 - 0.05, xi, y + H/2 + 0.05)
            prev_y = y

    # Convergence arrows to averaging box
    avg_y = -0.1
    avg_x = 7.0
    for xi in xs:
        ax.annotate('', xy=(avg_x, avg_y + 0.3), xytext=(xi, ys[-1] - H/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.0,
                                   connectionstyle='arc3,rad=0.1'))

    # Averaging box
    box(ax, avg_x, avg_y, 5.0, 0.6,
        'SOFT VOTING: Average Logits', '(mean of 3 model outputs)',
        'bilstm', fontsize=10, bold=True)

    # Output
    arrow(ax, avg_x, avg_y - 0.3, avg_x, avg_y - 0.75)
    box(ax, avg_x, avg_y - 1.0, 4.5, 0.5,
        'OUTPUT: Bipolar / Unipolar', '', 'output', fontsize=10, bold=True)

    # Param count boxes
    ax.text(7.0, 8.2, 'Ensemble CNN-LSTM\n3 × 223,682 = 671,046 total parameters',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#2C3E50',
            bbox=dict(facecolor='#FDEDEC', edgecolor='#E74C3C',
                      boxstyle='round,pad=0.4', linewidth=2.0))

    ax.set_title('Architecture 2d: Ensemble CNN-LSTM\n3 independent models trained with different random seeds → soft voting averages predictions',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    p = RESULTS_DIR / "figN_arch_ensemble.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE O: Classical ML Pipelines (XGBoost + Logistic Regression) ──────
def fig_classical_ml():
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.patch.set_facecolor(CLR['bg'])

    titles = [
        'Approach 3-1C: Logistic Regression\n5 Features  ·  LOOCV  ·  60.87% accuracy',
        'Approach 3-1B: XGBoost\n19 Features  ·  LOOCV  ·  39.13% accuracy',
    ]

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 6); ax.set_ylim(-0.5, 12.5)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    cx, W, H = 3.0, 4.4, 0.65

    # ── LOGISTIC REGRESSION pipeline (left) ──
    ax = axes[0]
    lr_layers = [
        (12.0, 'RAW ACTIVITY DATA',      '23 participants  ·  8-29 days each',   'input',   True),
        (11.0, 'Per-Participant\nZ-Score Normalization', 'Remove baseline differences', 'bn_relu', False),
        (10.0, 'Segment into Days',       '1,440 min = 24hr windows',              'pool',    False),
        ( 9.0, 'Feature 1: Daily Mean',   'Average activity across monitoring',    'conv',    False),
        ( 8.0, 'Feature 2: Day Variability','Std dev of daily means ← KEY feature', 'lstm',   True),
        ( 7.0, 'Feature 3: Day Range',    'Max − Min daily mean',                  'conv',    False),
        ( 6.0, 'Feature 4: Coeff of Var','Normalized variability',                'conv',    False),
        ( 5.0, 'Feature 5: Mean Daily Std','Within-day stability',                  'conv',    False),
        ( 3.8, 'Feature Vector',          '[1 × 5] per participant  (n=23 rows)',   'fc',      False),
        ( 2.8, 'Leave-One-Out CV',        'Train on 22, test on 1  ·  repeat ×23', 'dropout', False),
        ( 1.8, 'Logistic Regression',     'C=0.1, class_weight=balanced',          'bilstm',  True),
        ( 0.8, 'Softmax → Probability',   '',                                       'output',  False),
        ( 0.0, 'OUTPUT',                  'Bipolar / Unipolar  (60.87% LOOCV)',    'output',  True),
    ]
    prev_y = None
    for (y, lbl, sub, ck, bd) in lr_layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=8.5, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    # 5-feature brace
    ax.plot([5.55, 5.55], [4.6, 9.4], color=CLR['conv'][1], lw=2)
    ax.text(5.7, 7.0, '5\nfeatures\n(hand-\ncrafted)', va='center', fontsize=8,
            color=CLR['conv'][1], fontweight='bold')

    # ── XGBOOST pipeline (right) ──
    ax = axes[1]
    xgb_layers = [
        (12.0, 'RAW ACTIVITY DATA',       '23 participants  ·  8-29 days each',  'input',   True),
        (11.0, 'Per-Participant\nZ-Score Normalization','',                       'bn_relu', False),
        (10.2, 'Segment into Days',        '1,440 min windows',                   'pool',    False),
        ( 9.2, 'Group 1: Basic Stats',     'mean, std, max, min, IQR  (5 feat)',  'conv',    False),
        ( 8.2, 'Group 2: Day Variability', 'day_var, day_range, CV, daily_std  (4 feat)', 'lstm', True),
        ( 7.2, 'Group 3: Daily Stats',     'mean_daily_max, daily_max_var  (2 feat)',     'conv', False),
        ( 6.2, 'Group 4: Activity Dist.',  'low/high activity fraction  (2 feat)',         'conv', False),
        ( 5.2, 'Group 5: Autocorrelation', 'lag-1, lag-2 correlation  (2 feat)',           'conv', False),
        ( 4.2, 'Group 6: Advanced',        'trend, entropy, peaks, sleep_cycles  (4 feat)','conv', False),
        ( 3.2, 'Feature Vector',           '[1 × 19] per participant  (n=23 rows)',        'fc',   False),
        ( 2.2, 'Leave-One-Out CV',         'Train on 22, test on 1  ·  repeat ×23',       'dropout', False),
        ( 1.2, 'XGBoost Classifier',       'max_depth=3/5/7 (identical results!)',         'bilstm',  True),
        ( 0.4, 'Probability + Feature Imp.','',                                             'output',  False),
        ( 0.0, 'OUTPUT',                   'Bipolar / Unipolar  (39.13% LOOCV)',           'output',  True),
    ]
    prev_y = None
    for (y, lbl, sub, ck, bd) in xgb_layers:
        box(ax, cx, y, W, H, lbl, sub, ck, fontsize=8.5, bold=bd)
        if prev_y is not None:
            arrow(ax, cx, prev_y - H/2, cx, y + H/2)
        prev_y = y

    # 19-feature brace
    ax.plot([5.55, 5.55], [3.8, 9.6], color=CLR['conv'][1], lw=2)
    ax.text(5.7, 6.7, '19\nfeatures\n(6 groups)', va='center', fontsize=8,
            color=CLR['conv'][1], fontweight='bold')

    # Add result badge
    for ax_i, acc, col in [(axes[0], '60.87%', '#27AE60'), (axes[1], '39.13%', '#E74C3C')]:
        ax_i.text(0.5, 0.5, acc, ha='center', va='center', fontsize=14, fontweight='bold',
                  color=col, transform=ax_i.transAxes,
                  bbox=dict(facecolor='white', edgecolor=col,
                            boxstyle='round,pad=0.5', linewidth=2, alpha=0.0))

    plt.suptitle('Classical ML Pipelines (Approach 3)\nParticipant-Level Features → Leave-One-Out Cross-Validation',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = RESULTS_DIR / "figO_arch_classical_ml.png"
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── FIGURE P: All 5 Neural Architectures Side-by-Side ─────────────────────
def fig_all_comparison():
    """Compact comparison of all 5 neural architectures in one figure."""
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(CLR['bg'])

    cols = 5
    col_w = 22 / cols
    col_xs = [col_w * i + col_w / 2 for i in range(cols)]

    # Data for each architecture
    architectures = [
        {
            'title': 'Baseline\nCNN-LSTM',
            'subtitle': '223,682 params\nAcc: 63.4%\n★ BEST RELIABLE',
            'title_color': '#27AE60',
            'layers': [
                ('Input\n[B×1×1,440]',         'input'),
                ('Conv1D k=7\n64 filters',      'conv'),
                ('BN + ReLU\nMaxPool →720',     'bn_relu'),
                ('Conv1D k=5\n128 filters',     'conv'),
                ('BN + ReLU\nMaxPool →360',     'bn_relu'),
                ('Transpose\n[B×360×128]',      'dropout'),
                ('LSTM\nhidden=128',            'lstm'),
                ('Last h\n[B×128]',             'dropout'),
                ('FC 128→256\nReLU+Drop',       'fc'),
                ('FC 256→64\nReLU+Drop',        'fc'),
                ('Linear\n64→2',               'output'),
                ('Output\nBipolar/Uni',         'output'),
            ],
        },
        {
            'title': 'BiLSTM\n(Approach 2a)',
            'subtitle': '388,546 params\nAcc: 55.4%\nBidirectional',
            'title_color': '#E74C3C',
            'layers': [
                ('Input\n[B×1×1,440]',          'input'),
                ('Conv1D k=7\n64 filters',       'conv'),
                ('BN + ReLU\nMaxPool →720',      'bn_relu'),
                ('Conv1D k=5\n128 filters',      'conv'),
                ('BN + ReLU\nMaxPool →360',      'bn_relu'),
                ('Transpose\n[B×360×128]',       'dropout'),
                ('BiLSTM\n←128 + 128→',          'bilstm'),
                ('Last fwd+bwd\n[B×256]',        'bilstm'),
                ('FC 256→256\nReLU+Drop',        'fc'),
                ('FC 256→64\nReLU+Drop',         'fc'),
                ('Linear\n64→2',                'output'),
                ('Output\nBipolar/Uni',          'output'),
            ],
        },
        {
            'title': 'Attention\nLSTM (2b)',
            'subtitle': '223,811 params\nAcc: 48.2%\nAttention weights',
            'title_color': '#8E44AD',
            'layers': [
                ('Input\n[B×1×1,440]',          'input'),
                ('Conv1D k=7\n64 filters',       'conv'),
                ('BN + ReLU\nMaxPool →720',      'bn_relu'),
                ('Conv1D k=5\n128 filters',      'conv'),
                ('BN + ReLU\nMaxPool →360',      'bn_relu'),
                ('Transpose\n[B×360×128]',       'dropout'),
                ('LSTM\nhidden=128 (ALL h)',     'lstm'),
                ('Attention\nSoftmax weights',   'attn'),
                ('Weighted Sum\n[B×128]',        'attn'),
                ('FC 128→256\nReLU+Drop',        'fc'),
                ('FC 256→64\nReLU+Drop',         'fc'),
                ('Linear 64→2\nOutput',         'output'),
            ],
        },
        {
            'title': 'RNN-LSTM\n(Approach 2c)',
            'subtitle': '210,818 params\nAcc: 100% ⚠\n(Unstable)',
            'title_color': '#95A5A6',
            'layers': [
                ('Input\n[B×1×1,440]',          'input'),
                ('Transpose\n[B×1,440×1]',      'dropout'),
                ('SimpleRNN\n1→64, ReLU',       'rnn'),
                ('[B×1,440×64]\n',              'rnn'),
                ('SimpleRNN\n64→128, ReLU',     'rnn'),
                ('[B×1,440×128]\n',             'rnn'),
                ('LSTM\nhidden=128',            'lstm'),
                ('Last h\n[B×128]',             'dropout'),
                ('FC 128→256\nReLU+Drop',       'fc'),
                ('FC 256→64\nReLU+Drop',        'fc'),
                ('Linear\n64→2',               'output'),
                ('Output\n⚠ Unstable',         'output'),
            ],
        },
        {
            'title': 'Ensemble\n(Approach 2d)',
            'subtitle': '671,046 params\nAcc: 59.7%\nSoft voting',
            'title_color': '#2C3E50',
            'layers': [
                ('Input\n[B×1×1,440]',          'input'),
                ('CNN-LSTM\nModel 1 (seed=42)',  'conv'),
                ('CNN-LSTM\nModel 2 (seed=43)',  'conv'),
                ('CNN-LSTM\nModel 3 (seed=44)',  'conv'),
                ('Logits₁\n[B×2]',             'fc'),
                ('Logits₂\n[B×2]',             'fc'),
                ('Logits₃\n[B×2]',             'fc'),
                ('Soft Vote\n(avg logits)',      'bilstm'),
                ('\n',                           'dropout'),
                ('\n',                           'dropout'),
                ('\n',                           'dropout'),
                ('Output\nBipolar/Uni',         'output'),
            ],
        },
    ]

    # Axes per column
    axs = []
    for i in range(cols):
        ax = fig.add_axes([i / cols + 0.01, 0.02, 1/cols - 0.02, 0.88])
        ax.set_xlim(0, 4); ax.set_ylim(-0.3, 13.2)
        ax.axis('off')
        axs.append(ax)

    cx = 2.0; W = 3.4; H = 0.72
    ys = np.linspace(12.5, 0.2, 12)

    for ax, arch in zip(axs, architectures):
        # Title block
        ax.text(cx, 12.95, arch['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=arch['title_color'],
                bbox=dict(facecolor='#F8F9FA', edgecolor=arch['title_color'],
                          boxstyle='round,pad=0.35', linewidth=2.5))
        ax.text(cx, 11.85, arch['subtitle'], ha='center', va='center',
                fontsize=8.5, color='#555555', style='italic',
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.25', linewidth=1))

        prev_y = None
        for (lbl, ck), y in zip(arch['layers'], ys):
            if lbl.strip():
                box(ax, cx, y, W, H - 0.05, lbl, '', ck, fontsize=8)
            if prev_y is not None and lbl.strip():
                arrow(ax, cx, prev_y - H/2, cx, y + H/2)
            if lbl.strip():
                prev_y = y

    # Column headers (shared title area)
    fig.text(0.5, 0.97,
             'All Neural Network Architectures — Side-by-Side Comparison\n'
             'Bipolar vs. Unipolar Depression Classification from Wrist Actigraphy',
             ha='center', va='center', fontsize=15, fontweight='bold', color='#2C3E50')

    # Legend bar at bottom
    legend_items = [
        ('Input / Reshape',      CLR['input'][0],   CLR['input'][1]),
        ('Conv1D',               CLR['conv'][0],    CLR['conv'][1]),
        ('BatchNorm + ReLU',     CLR['bn_relu'][0], CLR['bn_relu'][1]),
        ('MaxPool',              CLR['pool'][0],    CLR['pool'][1]),
        ('LSTM / RNN',           CLR['lstm'][0],    CLR['lstm'][1]),
        ('BiLSTM',               CLR['bilstm'][0],  CLR['bilstm'][1]),
        ('Attention',            CLR['attn'][0],    CLR['attn'][1]),
        ('Fully Connected',      CLR['fc'][0],      CLR['fc'][1]),
        ('Output / Softmax',     CLR['output'][0],  CLR['output'][1]),
    ]
    patches = [mpatches.Patch(facecolor=fc, edgecolor=ec, label=lbl, linewidth=1.5)
               for lbl, fc, ec in legend_items]
    fig.legend(handles=patches, loc='lower center', ncol=9,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    p = RESULTS_DIR / "figP_arch_all_comparison.png"
    plt.savefig(p, dpi=180, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"[SAVED] {p}")


# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("ARCHITECTURE DIAGRAMS: Generating all model diagrams")
    print("=" * 65)

    print("\n[J/7] Baseline CNN-LSTM (detailed)...")
    fig_cnnlstm()

    print("[K/7] Bidirectional LSTM (detailed)...")
    fig_bilstm()

    print("[L/7] Attention LSTM (detailed)...")
    fig_attention()

    print("[M/7] RNN-LSTM (detailed)...")
    fig_rnnlstm()

    print("[N/7] Ensemble CNN-LSTM...")
    fig_ensemble()

    print("[O/7] Classical ML pipelines (XGBoost + LogReg)...")
    fig_classical_ml()

    print("[P/7] All architectures side-by-side comparison...")
    fig_all_comparison()

    print("\n" + "=" * 65)
    print("ALL ARCHITECTURE DIAGRAMS COMPLETE")
    print("=" * 65)
    print("\nFiles saved to results/:")
    for p in sorted(Path("results").glob("fig[J-P]*.png")):
        print(f"  {p}")
    print("\nFor poster:")
    print("  figP — All 5 architectures side-by-side  (center/methods panel)")
    print("  figJ — Detailed CNN-LSTM baseline         (methods detail)")
    print("  figO — Classical ML pipelines             (Approach 3 panel)")
