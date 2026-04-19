import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 14))
fig.suptitle('Approach 3: Architecture Comparison (1A vs 1B vs 1C)', fontsize=20, fontweight='bold', y=0.98)

# Define colors
color_input = '#E8F4F8'
color_process = '#B8E6F0'
color_model = '#5DADE2'
color_output = '#A9DFBF'

# Fixed spacing parameters
BOX_HEIGHT = 0.8
BOX_WIDTH = 8.5
ARROW_LENGTH = 0.8  # Space between boxes
CENTER_X = 5
MARGIN_X = 0.75

def draw_box(ax, x, y, width, height, text, color, fontsize=9):
    """Draw a box at position (x, y) with given text"""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.08",
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if color != color_process else 'normal')

def draw_arrow(ax, x, y_from, y_to):
    """Draw arrow between boxes"""
    arrow = FancyArrowPatch((x, y_from), (x, y_to),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
    ax.add_patch(arrow)

# ============ APPROACH 1A: Multi-Scale Windows ============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')
ax.set_title('1A: Multi-Scale CNN-LSTM\n(Neural Network)', fontsize=13, fontweight='bold', pad=15)

# Fixed y-positions from top to bottom
y_positions = [23, 21, 19, 17, 15, 13, 11, 9, 7]

# Input
draw_box(ax, CENTER_X, y_positions[0], BOX_WIDTH, BOX_HEIGHT,
         'Raw Activity Data\n(24hr / 48hr / 72hr)', color_input, fontsize=9)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[0] - BOX_HEIGHT/2 - 0.1, y_positions[1] + BOX_HEIGHT/2 + 0.1)

# CNN Block 1
draw_box(ax, CENTER_X, y_positions[1], BOX_WIDTH, BOX_HEIGHT,
         'CNN Block 1: Conv1D\n(64 filters, k=7)', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[1] - BOX_HEIGHT/2 - 0.1, y_positions[2] + BOX_HEIGHT/2 + 0.1)

# CNN Block 2
draw_box(ax, CENTER_X, y_positions[2], BOX_WIDTH, BOX_HEIGHT,
         'CNN Block 2: Conv1D\n(128 filters, k=5)', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[2] - BOX_HEIGHT/2 - 0.1, y_positions[3] + BOX_HEIGHT/2 + 0.1)

# LSTM
draw_box(ax, CENTER_X, y_positions[3], BOX_WIDTH, BOX_HEIGHT,
         'LSTM (128 units)', color_model, fontsize=9)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[3] - BOX_HEIGHT/2 - 0.1, y_positions[4] + BOX_HEIGHT/2 + 0.1)

# FC Layers
draw_box(ax, CENTER_X, y_positions[4], BOX_WIDTH, BOX_HEIGHT,
         'FC: 256 -> 64 -> 2\n(dropout=0.4)', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[4] - BOX_HEIGHT/2 - 0.1, y_positions[5] + BOX_HEIGHT/2 + 0.1)

# Output
draw_box(ax, CENTER_X, y_positions[5], BOX_WIDTH, BOX_HEIGHT,
         'Binary Classification', color_output, fontsize=9)

# ============ APPROACH 1B: Feature Engineering + XGBoost ============
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')
ax.set_title('1B: Feature Eng. + XGBoost\n(Ensemble Learning)', fontsize=13, fontweight='bold', pad=15)

# Input
draw_box(ax, CENTER_X, y_positions[0], BOX_WIDTH, BOX_HEIGHT,
         'Raw Activity Data\n(All windows)', color_input, fontsize=9)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[0] - BOX_HEIGHT/2 - 0.1, y_positions[1] + BOX_HEIGHT/2 + 0.1)

# Feature Engineering
draw_box(ax, CENTER_X, y_positions[1], BOX_WIDTH, BOX_HEIGHT,
         'Hand-Crafted Features (19)\nVariability, Entropy, Autocorr', color_process, fontsize=7.5)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[1] - BOX_HEIGHT/2 - 0.1, y_positions[2] + BOX_HEIGHT/2 + 0.1)

# Feature Ranking
draw_box(ax, CENTER_X, y_positions[2], BOX_WIDTH, BOX_HEIGHT,
         'Feature Ranking\nby Importance', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[2] - BOX_HEIGHT/2 - 0.1, y_positions[3] + BOX_HEIGHT/2 + 0.1)

# XGBoost
draw_box(ax, CENTER_X, y_positions[3], BOX_WIDTH, BOX_HEIGHT,
         'XGBoost Ensemble\nmax_depth=[3,5,7], n_est=100', color_model, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[3] - BOX_HEIGHT/2 - 0.1, y_positions[5] + BOX_HEIGHT/2 + 0.1)

# Output
draw_box(ax, CENTER_X, y_positions[5], BOX_WIDTH, BOX_HEIGHT,
         'Classification +\nFeature Importance', color_output, fontsize=9)

# ============ APPROACH 1C: Participant Aggregation + Logistic Regression ============
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.axis('off')
ax.set_title('1C: Participant Agg. + LogReg\n(Simple Statistical)', fontsize=13, fontweight='bold', pad=15)

# Input
draw_box(ax, CENTER_X, y_positions[0], BOX_WIDTH, BOX_HEIGHT,
         'Raw Activity Data\n(Per participant)', color_input, fontsize=9)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[0] - BOX_HEIGHT/2 - 0.1, y_positions[1] + BOX_HEIGHT/2 + 0.1)

# Aggregation
draw_box(ax, CENTER_X, y_positions[1], BOX_WIDTH, BOX_HEIGHT,
         'Participant-Level Aggregation\n(One row per participant)', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[1] - BOX_HEIGHT/2 - 0.1, y_positions[2] + BOX_HEIGHT/2 + 0.1)

# Feature Summary
draw_box(ax, CENTER_X, y_positions[2], BOX_WIDTH, BOX_HEIGHT,
         '5 Statistical Features\n(Mean, Variance, Range, CoV)', color_process, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[2] - BOX_HEIGHT/2 - 0.1, y_positions[3] + BOX_HEIGHT/2 + 0.1)

# Logistic Regression
draw_box(ax, CENTER_X, y_positions[3], BOX_WIDTH, BOX_HEIGHT,
         'Logistic Regression\nC=[0.1, 1.0, 10.0]', color_model, fontsize=8)

# Arrow
draw_arrow(ax, CENTER_X, y_positions[3] - BOX_HEIGHT/2 - 0.1, y_positions[5] + BOX_HEIGHT/2 + 0.1)

# Output
draw_box(ax, CENTER_X, y_positions[5], BOX_WIDTH, BOX_HEIGHT,
         'Classification +\nLinear Weights', color_output, fontsize=9)

plt.tight_layout()
plt.savefig('results/figP_approach3_1A_1B_1C_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Fixed architecture comparison diagram saved!")
plt.close()

# Create a summary comparison table
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('tight')
ax.axis('off')

# Table data
table_data = [
    ['Aspect', '1A: Multi-Scale CNN-LSTM', '1B: Feature Eng. + XGBoost', '1C: Participant Agg. + LogReg'],
    ['Input Type', 'Raw time series\n(windows)', 'Raw time series\n(all windows)', 'Raw time series\n(per participant)'],
    ['Processing', 'CNN (64->128 filters)\nLSTM sequence modeling\nFC layers (256->64->2)', 'Hand-crafted features (19)\nEntropy, autocorrelation\nFragmentation, cycles', 'Statistical aggregation\nMean, variability\nRange, coef of var, stability'],
    ['Model Type', 'Deep Neural Network\n(223,682 parameters)', 'Gradient Boosting Ensemble\n(Decision trees)', 'Linear Classifier\n(Logistic Regression)'],
    ['Validation', 'Standard train/val/test split\n(window-level)', 'Leave-One-Out CV\n(participant-level)', 'Leave-One-Out CV\n(participant-level)'],
    ['Interpretability', '[Limited] Black box\n(Difficult to explain)', '[Good] Feature importance\n(Interpretable)', '[Excellent] Linear weights\n(Highly interpretable)'],
    ['Best Accuracy', '63.4% (48hr window)', '39.13% LOOCV', '60.87% LOOCV'],
    ['Complexity', 'High\n(Deep learning)', 'Medium\n(Ensemble)', 'Low\n(Simple)'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.28, 0.28, 0.28])

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 2.8)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#5DADE2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F4F8')
        else:
            table[(i, j)].set_facecolor('#F0F0F0')

plt.title('Approach 3 Experiments: Detailed Comparison', fontsize=14, fontweight='bold', pad=20)
plt.savefig('results/figP_approach3_comparison_table.png', dpi=300, bbox_inches='tight')
print("[OK] Comparison table saved!")
plt.close()

print("\n[OK] All diagrams fixed and saved successfully!")

