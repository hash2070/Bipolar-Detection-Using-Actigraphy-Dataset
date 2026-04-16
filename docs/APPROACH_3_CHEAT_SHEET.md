# BIPOLAR DETECTION: ONE-PAGE CHEAT SHEET

## THE PROBLEM

**Status**: Model predicts 100% unipolar despite:
- ✗ Perfectly balanced training data (1:1 downsampling)
- ✗ 4 different architectures (BiLSTM, Attention, RNN, Ensemble)  
- ✗ SMOTE synthetic data oversampling
- ✗ Weighted loss functions

**Why**: Only 8 bipolar subjects. Window size too small to capture bipolar's multi-day cycling.

---

## THE ROOT CAUSE (Pick One)

| Cause | Evidence | Solution |
|-------|----------|----------|
| **1. Data Too Small** | 8 bipolar = 1000 params/subject | Aggregate to participant level (1C) |
| **2. Window Size Wrong** | 24hr too short to see cycling | Try 48hr/72hr/7day windows (1A) |
| **3. Raw Sequences Weak** | Prior work: F1=0.82 with features vs your 0.52 | Extract hand-crafted features (1B) |
| **4. Signal Doesn't Exist** | Maybe bipolar/unipolar indistinguishable in this dataset | Run statistical tests (3A) |

---

## THE THREE APPROACHES (Ranked by Do-This-First)

### 🥇 1C: PARTICIPANT AGGREGATION (START HERE)

**What**: Compute variability per person. Train logistic regression on 23 people.

**Code**: 
```python
# For each person: compute day-to-day activity variability
variability = std(daily_activity_means)
# Bipolar: HIGH variability (mood cycling)
# Unipolar: LOW variability (sustained low)
# Train: LogisticRegression(variability → bipolar?)
```

**Time**: 3-4 hours  
**Expected**: 60-70% accuracy + crystal-clear interpretation  
**When Done**: EOD Apr 17

---

### 🥈 1A: MULTI-SCALE WINDOWS (IF 1C WORKS)

**What**: Train 5 models on windows: 6hr, 24hr, 48hr, 72hr, 7day

**Code**:
```python
for window_size in [360, 1440, 2880, 4320, 10080]:  # 6hr to 7day
    model = train_existing_cnn_lstm(window_size)
    evaluate(model)
# Expect: 48hr and 72hr >> 24hr
```

**Time**: 6 hours code + 2-3 hours training overnight  
**Expected**: 48hr model 70-75%, visual proof of temporal resolution  
**When Done**: EOD Apr 18

---

### 🥉 1B: FEATURE ENGINEERING (IF BOTH FAIL)

**What**: Extract 20+ features (variability, entropy, fragmentation, trends). Train XGBoost.

**Code**:
```python
features = {
    'day_to_day_variability': std(daily_means),
    'entropy': Shannon_entropy(activity_dist),
    'activity_range': max - min,
    'autocorrelation_48hr': corr(day_t, day_t+2),
    # ... 16 more features
}
model = XGBoost(features → bipolar?)
```

**Time**: 8 hours  
**Expected**: 65-75% + SHAP interpretability  
**When Done**: EOD Apr 19

---

## BACKUP PLAN (If All Fail)

### 3A: STATISTICAL SIGNIFICANCE TEST

**What**: T-test bipolar vs unipolar variability. Report Cohen's d.

**Result**: 
- If p < 0.05 + d > 0.5: "Signal exists, approach X captures it"
- If p > 0.05: "No statistical difference → n=8 too small"

**Time**: 2-3 hours  
**Value**: Scientifically honest framing for report

---

## DECISION TREE

```
START: Do you have 4 uninterrupted hours today?
│
├─ YES → Start 1C (Participant aggregation)
│        By evening: know if variability separates bipolar/unipolar
│        By tomorrow: either celebrate or pivot to 1A
│
└─ NO → Start 1B (Feature extraction)
        More modular, work in 2-hour chunks
        OR wake up early on Apr 17 and do 1C
```

---

## SUCCESS METRICS

| Approach | Success Threshold | Stretch Goal |
|----------|-------------------|--------------|
| **1C: Aggregation** | ≥ 65% LOOCV accuracy | ≥ 70% |
| **1A: Multi-scale** | 48hr model ≥ 70% accuracy | 75%+ |
| **1B: Features** | ≥ 65% test accuracy | ROC-AUC ≥ 0.70 |
| **3A: Statistics** | p < 0.05 in t-test | Cohen's d ≥ 0.5 |

---

## TIMELINE

```
Apr 17: 1C implementation + results
Apr 18: 1A training (overnight) + evaluation
Apr 19: 1B or 3A + decide on best approach
Apr 20: Final report + presentation prep
Apr 21: Video + poster + DONE
```

---

## WHAT TO WRITE IN FINAL REPORT

### If One Approach Works (65%+ accuracy):
> "We implemented Approach X, which achieved 70% test ROC-AUC by directly measuring bipolar's signature (multi-day variability / temporal dynamics). This demonstrates that bipolar/unipolar distinction is possible, but requires capturing temporal patterns at multi-day scale rather than single-day activity snapshots."

### If All Fail Around 65%:
> "Multiple approaches converged on 60-65% accuracy, suggesting the actigraphy signal is near our current detection ceiling with n=8 bipolar subjects. Statistical analysis confirms the distinction exists (p<0.05) but is subtle (Cohen's d=0.3). Future work requires larger cohorts (n=50+ bipolar) or alternative biomarkers."

### If Signal Doesn't Exist Statistically:
> "Statistical analysis revealed no significant difference in activity variability between bipolar and unipolar depressed patients (p=0.12), suggesting actigraphy patterns are indistinguishable for these phenotypes during acute depressive episodes. This negative result has clinical value: ruling out passive monitoring as a bipolar detection method in small samples."

---

## FILES YOU NOW HAVE

1. **APPROACH_3_EXECUTIVE_SUMMARY.md** ← READ THIS FIRST (10 min)
2. **APPROACH_3_QUICK_REFERENCE.md** ← Quick decision guide
3. **APPROACH_3_NEW_STRATEGIES.md** ← Detailed explanation of all 9 approaches
4. **APPROACH_3_CODE_TEMPLATES.md** ← Copy-paste ready code
5. **APPROACH_3_CHEAT_SHEET.md** ← This file

---

## COPY-PASTE TO START

**1C Implementation**:
```bash
# Create classify_by_variability.py with code from APPROACH_3_CODE_TEMPLATES.md
python classify_by_variability.py
# Should finish in 30 min + show LOOCV accuracy
```

**1A Implementation**:
```bash
# Create train_exp2_multiscale.py with code from templates
python train_exp2_multiscale.py
# Train 5 models overnight, evaluate tomorrow
```

**1B Implementation**:
```bash
# Create extract_features.py and train_exp2_xgboost.py
python extract_features.py
python train_exp2_xgboost.py
# Get results in 2-3 hours
```

---

## KEY INSIGHT

**This isn't a model problem. It's a signal problem.**

- Architecture: ✓ Already optimized (CNN-LSTM appropriate)
- Data balance: ✓ Already tried (downsampling doesn't help)
- Class weighting: ✓ Already tried (loss functions don't help)
- Sample size: ✗ Core issue (8 bipolar insufficient for deep learning)

**Solution**: Don't fight the constraints. Embrace them.
- Use shallow models (logistic regression, XGBoost) suited to small n
- Look for explicit signals (variability, entropy, trends) not implicit ones
- Aggregate to participant level (n=23) not window level (n=6000)

---

## GO-NO GO DECISION POINT: Apr 18 EOD

**After trying 1C:**

**GO**: "Variability shows 65%+ LOOCV accuracy"
→ Use 1C results in final report
→ Run 1A as supporting evidence
→ Submit with confidence

**NO-GO**: "Variability shows <55% accuracy, no better than majority class"
→ Immediately pivot to 1A (multi-scale windows)
→ By EOD Apr 18, will have 1A results
→ Probably better (windows address signal problem directly)

**ALL-NO-GO**: Both 1C and 1A fail
→ Run 1B (features) on Apr 19
→ Worst case: statistical tests + honest assessment

---

## YOU ARE HERE

```
Apr 9:  Results from Approach 1 & 2 (both failed)
Apr 16: Analysis complete → 3 new strategies ranked
Apr 17: [YOU ARE HERE] → Pick 1C/1A/1B, start coding
Apr 21: Final submission
```

**Status**: You have a clear path. Execution is next.

**TL;DR**: Start with 1C today. 3-4 hours. Results by evening. Know by tomorrow if approach has merit.

