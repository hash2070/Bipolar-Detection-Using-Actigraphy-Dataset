# Final Results Summary: Bipolar Detection Via Actigraphy

**Project**: Sequence Modeling of Wrist-Worn Actigraphy for Differentiating Bipolar and Unipolar Depressive Episodes
**Authors**: Shikha Masurkar, Harsh Mukesh Sharma
**Course**: CSCI 5922, CU Boulder
**Date**: April 9, 2026

---

## Executive Summary

We developed a 1D-CNN-LSTM architecture to distinguish bipolar from unipolar depression using raw wrist actigraphy data. Two controlled experiments reveal:

1. **Experiment 1 (Healthy vs. Depressed)**: Model achieves ~55% ROC-AUC, underperforming hand-crafted feature baselines (F1=0.82)
2. **Experiment 2 (Bipolar vs. Unipolar)**: High nominal accuracy (88%) but model learns only majority class prediction due to extreme imbalance

**Key Insight**: End-to-end learning on raw sequences is more challenging than prior work with engineered features, suggesting raw actigraphy requires preprocessing or architectural innovations not present in our design.

---

## Experiment 1: Healthy vs. Depressed

### Setup
- **Participants**: All 55 (32 healthy, 23 depressed)
- **Data**: 24,901 windows (1,440 minutes each)
- **Split**: 44 train / 5 val / 6 test participants
- **Class Balance**: 16,248 healthy vs 8,653 depressed (1.88:1 ratio)
- **Training**: Class-weighted cross-entropy, early stopping (patience=10)

### Results
| Metric | Value |
|--------|-------|
| **Accuracy** | 53.42% |
| **Precision** | 59.02% |
| **Recall** | 45.78% |
| **F1-Score** | 51.57% |
| **ROC-AUC** | 55.62% |

### Analysis

The model barely outperforms random guessing (50%). While class weighting improved recall (45% vs 16% without weights), the overall discrimination is poor.

**Possible causes:**
1. **Raw sequences lack signal**: Hand-crafted features (mean activity, variance, zero-crossing rate) may be more informative for this task
2. **Architecture mismatch**: CNN designed for images/sequences may not be optimal for 1D activity time series
3. **Insufficient normalization**: Z-score per participant may not adequately remove inter-subject baseline differences
4. **Temporal window size**: 24-hour windows may not capture relevant circadian disruptions

**Comparison to prior work:**
- Prior baseline (DNN + SMOTE with hand-crafted features): **F1 = 0.82**
- Our approach (1D-CNN-LSTM, raw sequences): **F1 = 0.52**
- **Delta**: -0.30 (our end-to-end approach underperforms by 37%)

---

## Experiment 2: Bipolar vs. Unipolar

### Setup
- **Participants**: Only 23 condition (8 bipolar, 15 unipolar)
- **Data**: 8,653 windows total
- **Split**: 18 train / 2 val / 3 test participants
- **Class Balance**: 3,044 bipolar vs 5,609 unipolar (1.84:1 ratio)
- **Imbalance Handling**:
  - SMOTE on training set (window-level oversampling in CNN feature space)
  - Weighted cross-entropy loss
- **Training**: Early stopping triggered at epoch 18 due to validation loss divergence

### Results
| Metric | Value |
|--------|-------|
| **Accuracy** | 88.01% |
| **Precision** | 100.00% |
| **Recall** | 88.01% |
| **F1-Score** | 93.62% |
| **ROC-AUC** | NaN (undefined) |

### Confusion Matrix
```
                Predicted
              Bipolar  Unipolar
Actual Bipolar    0        0
       Unipolar  121      888
```

### Critical Analysis

**The apparent success is illusory:**
The model predicts **zero bipolar cases** in the test set. All 1,009 test samples are classified as unipolar. This explains the unintuitive metrics:
- **High accuracy (88%)**: Because unipolar is the majority class (888/1,009 = 88%)
- **100% precision**: No false positives (can't have false positives if you never predict bipolar)
- **ROC-AUC undefined**: Cannot compute ROC curve when one class has zero predicted samples

**Root causes:**
1. **Extreme class imbalance in test set**: Only ~100 bipolar windows expected in test set vs 900 unipolar
2. **Tiny number of bipolar participants in test (1-2)**: With participant-level splits, test set likely contains few bipolar subjects
3. **Model collapse**: Despite SMOTE on training set, the model learned to ignore bipolar features entirely
4. **Validation instability**: Val accuracy wildly fluctuates (0.35 to 0.99), suggesting overfitting to tiny val set (1,008 windows across 2 participants)

---

## Key Learnings & Limitations

### What Worked
✓ Clean data pipeline with participant-level stratification (prevents leakage)
✓ Appropriate use of SMOTE and class weighting for imbalanced data
✓ Early stopping prevented unbounded overfitting
✓ Architectural implementation matches paper design (223K parameters)

### What Didn't Work
✗ Raw sequence learning underperformed engineered features
✗ Class imbalance too severe for model to learn minority class
✗ Validation set too small for reliable hyperparameter selection
✗ Temporal window size may not match clinically relevant patterns

### Fundamental Limitations
1. **Sample size**: 8 bipolar participants is too small for deep learning (typically need 100+ per class)
2. **Participant-level splits**: Ensures no leakage but creates tiny test sets (1-2 participants per class)
3. **Window-level sampling**: Inflates apparent dataset size but doesn't add independent information
4. **Clinical phenotype overlap**: Bipolar and unipolar depression look similar during depressed episodes

---

## Recommendations for Future Work

1. **Collect more data**: Current dataset insufficient. Need 50+ bipolar participants minimum.

2. **Explore hand-crafted features**: Prior baseline was more successful—consider:
   - Mean activity per hour
   - Inter-day variability in activity timing
   - Fragmentation indices (sleep disruption)
   - Spectral analysis of circadian rhythms

3. **Architectural improvements**:
   - Attention mechanisms to weight important time windows
   - Separate branches for short-term (CNN) and long-term (LSTM) patterns
   - Multi-task learning with MADRS depression scores as auxiliary task

4. **Preprocessing innovations**:
   - Activity thresholding (separate locomotive vs. non-locomotive activity)
   - Circadian extraction (isolate periodic components)
   - Adaptive normalization (preserve relative intensity structure)

5. **Methodological rigor**:
   - Leave-one-participant-out cross-validation for stable estimates
   - Stratified by both class AND participant characteristics (age, sex)
   - Compare against clinical baseline (clinician predictions from actigraphy)

---

## Conclusion

This work demonstrates that distinguishing bipolar from unipolar depression via passive actigraphy is **clinically important but technically challenging**. Our end-to-end deep learning approach did not outperform engineered feature baselines, suggesting:

- Raw motor activity sequences require preprocessing or feature extraction to isolate discriminative patterns
- Classification requires methods suited to small sample sizes (not deep neural networks)
- Future progress requires either larger datasets or domain-informed feature engineering

Despite limited success, the **project framework is sound** and could serve as a foundation for future work with more data. The participant-level stratified split, SMOTE handling, and multi-stage experimental design are reproducible and generalizable.

---

## References

1. Garcia-Ceja, E., et al. (2018). Depresjon: A motor activity database of depression episodes. *ACM Multimedia Systems Conference*.
2. Jakobsen, P., et al. (2020). Applying machine learning in motor activity time series of depressed bipolar and unipolar patients. *PLOS ONE*, 15(8).
3. Wehr, T.A., et al. (1987). 48-hour sleep-wake cycles in manic-depressive illness. *Archives of General Psychiatry*, 39(5).
