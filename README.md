# Bipolar Detection Using Actigraphy Dataset

**Course**: CSCI 5922 - Deep Learning & Neural Networks, CU Boulder
**Authors**: Shikha Masurkar, Harsh Mukesh Sharma

## Project Overview

This project explores whether continuous wrist-worn actigraphy data can distinguish between Bipolar Disorder and Major Depressive Disorder (MDD). Bipolar patients are frequently misdiagnosed as MDD, leading to harmful treatment with antidepressants that can trigger manic episodes. We develop a hybrid 1D-CNN-LSTM architecture that learns directly from raw minute-level activity counts without manual feature engineering.

## Key Contributions

- **Two-stage experimental design**:
  - Exp 1: Healthy vs. Depressed baseline (validates architecture)
  - Exp 2: Bipolar vs. Unipolar (core clinical challenge)
- **End-to-end deep learning**: No hand-crafted features, raw actigraphy sequences
- **Multi-scale temporal modeling**: CNN captures short-term kinetic patterns; LSTM tracks long-range circadian disruptions
- **Rigorous evaluation**: Participant-level stratified splits prevent data leakage; SMOTE addresses class imbalance in Exp 2

## Dataset: Depresjon

- **Source**: Simula Research Laboratory
- **Participants**: 55 total (32 healthy controls, 23 mood disorder patients)
  - 8 Bipolar (Bipolar I/II)
  - 15 Unipolar (Major Depressive Disorder)
- **Collection**: Actiwatch (32 Hz accelerometer, aggregated to 1-min epochs)
- **Duration**: ~12.7 days per participant (5-20 days range)
- **Metadata**: Demographics, MADRS depression ratings, inpatient/outpatient status

## Model Architecture

```
Input: 1440 × 1 (24-hr activity window)
    ↓
CNN Block 1: Conv1D(64, k=7) + BatchNorm + ReLU + MaxPool
    → 720 × 64
CNN Block 2: Conv1D(128, k=5) + BatchNorm + ReLU + MaxPool
    → 360 × 128
    ↓
LSTM (128 hidden units)
    → 128 (final hidden state)
    ↓
FC1: 128 → 256 (ReLU + Dropout 0.4)
    ↓
FC2: 256 → 64 (ReLU + Dropout 0.4)
    ↓
Output: 64 → num_classes (Softmax)
```

**Total Parameters**: 223,682

## Project Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Data loading, preprocessing, participant-level stratified splits |
| `model.py` | 1D-CNN-LSTM architecture definition |
| `train_exp1.py` | Experiment 1 training pipeline (healthy vs. depressed) |
| `train_exp2.py` | Experiment 2 training pipeline with SMOTE (bipolar vs. unipolar) |
| `visualize.py` | Visualization utilities (ROC curves, confusion matrices) |
| `run_all.py` | Master runner script for both experiments |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

The `depresjon.zip` file should be extracted to create:
```
depresjon/
├── data/
│   ├── scores.csv
│   ├── condition/ (23 participant CSVs)
│   └── control/   (32 participant CSVs)
```

The data loader works directly with the extracted folder structure.

### Running Experiments

**Option 1: Run individual experiments**
```bash
python train_exp1.py  # Healthy vs. Depressed
python train_exp2.py  # Bipolar vs. Unipolar (with SMOTE)
```

**Option 2: Run both experiments with visualizations**
```bash
python run_all.py
```

### Results

Results are saved to `results/`:
- `exp1_results.json` / `exp2_results.json` - Metrics (accuracy, precision, recall, F1, ROC-AUC)
- `exp1_y_true.npy`, `exp1_y_pred.npy`, `exp1_y_probs.npy` - Predictions for visualization
- `exp*_roc_curve.png` - ROC curve plots
- `exp*_confusion_matrix.png` - Confusion matrix heatmaps

## Experimental Design

### Experiment 1: Healthy vs. Depressed

- **Task**: Binary classification (healthy/depressed)
- **Participants**: All 55 (44 train, 5 val, 6 test)
- **Windows**: 24,901 total (20,385 train, 2,326 val, 2,190 test)
- **Purpose**: Validate architecture and compare against prior baselines
- **Prior baseline**: DNN + SMOTE achieved F1=0.82

### Experiment 2: Bipolar vs. Unipolar (Core Challenge)

- **Task**: Differential diagnosis within depressed patients
- **Participants**: Only 23 condition (15 unipolar, 8 bipolar)
- **Class imbalance**: 8 bipolar vs 15 unipolar (1:1.875 ratio)
- **Handling imbalance**:
  - SMOTE on training set (window-level oversampling)
  - Weighted cross-entropy loss (inverse class frequency)
- **Primary metric**: ROC-AUC (robust to imbalance)
- **Purpose**: Determine if temporal structure provides diagnostic signal

## Key Design Decisions

1. **Participant-level stratified split**: Prevents data leakage (no participant in both train & test)
2. **Z-score normalization per participant**: Removes inter-subject baseline differences
3. **24-hr windows with 60-min stride**: Consistent daily-scale context
4. **CNN→LSTM stack**:
   - CNN detects local kinetic patterns (restless activity, sleep disruption)
   - LSTM tracks multi-day circadian phase shifts
5. **SMOTE at window level** (Exp 2): Balances training after CNN feature extraction

## Expected Results

**Experiment 1** should outperform prior baselines:
- Prior DNN+SMOTE: F1=0.82
- Target: F1 ≥ 0.85 (end-to-end learning advantage)

**Experiment 2** is the real challenge due to:
- Small sample size (8 bipolar)
- Subtle phenotypic differences during depressed state
- Realistic clinical scenario

## References

1. Garcia-Ceja, E., et al. (2018). Depresjon: A motor activity database of depression episodes. ACM Multimedia Systems.
2. Zanella-Calzada, L.A., et al. (2019). Feature extraction in motor activity signal. Diagnostics.
3. Boudebesse, C., et al. (2014). Correlations between sleep-wake rhythms and mood in bipolar disorder. European Psychiatry.
4. Wehr, T.A., et al. (1987). 48-hour sleep-wake cycles in manic-depressive illness. Archives of General Psychiatry.
