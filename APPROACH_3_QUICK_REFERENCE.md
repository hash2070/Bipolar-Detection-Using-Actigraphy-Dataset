# QUICK REFERENCE: Ranked Recommendations for Bipolar vs Unipolar Detection

**Deadline**: April 21, 2026 (5 days)  
**Goal**: Improve from 65% accuracy / NaN ROC-AUC baseline

---

## Top 3 Recommendations (Ranked)

### 🥇 **RANK 1: Participant-Level Aggregation (1C)**

**Why First**: Fastest to implement, most interpretable, aligns with clinical reality

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Feasibility** | ⭐⭐⭐⭐⭐ (Very Easy) | 3-4 hours coding |
| **Likelihood of Success** | ⭐⭐⭐⭐ (70% confidence) | Targets bipolar phenotype directly |
| **Code Complexity** | Low | Pandas aggregation + sklearn |
| **Scientific Insight** | ⭐⭐⭐⭐ (Very High) | Answers: does variability matter? |
| **Time to Results** | 1 day | Can validate by EOD Apr 17 |
| **Reportability** | Excellent | Easy to explain, no black boxes |

**What It Does**:
```
Aggregate each participant's 10-14 day activity into statistics:
  - Mean daily activity
  - Day-to-day variability (the key bipolar signal)
  - Activity range (max - min)
  - Coefficient of variation
  
Train simple model: LogisticRegression(features) → bipolar vs unipolar
Use LOOCV for 23 participants
Expected: 60-70% accuracy
```

**Implementation Plan**:
```python
# 1. data_loader.py: add method to compute participant statistics
# 2. Create classify_by_variability.py:
#    - Load metadata + activity data
#    - Compute variability metrics per participant
#    - Train sklearn LogisticRegression
#    - Report LOOCV scores
# 3. Visualize: variability vs diagnosis scatter plot
# 4. Time: 3-4 hours
```

**Success Criteria**:
- LOOCV accuracy ≥ 65%
- Variability coefficient correlates with bipolar (p < 0.05)
- Can interpret results in 1 sentence

**If It Fails**: 
- Suggests variability alone insufficient
- Pivot to 1B (need more features) or 1A (need longer windows)

---

### 🥈 **RANK 2: Multi-Scale Temporal Windows (1A)**

**Why Second**: Most scientifically grounded, directly addresses window size problem

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Feasibility** | ⭐⭐⭐⭐ (Easy-Medium) | 4-6 hours coding |
| **Likelihood of Success** | ⭐⭐⭐⭐⭐ (90% confidence) | Strong clinical rationale |
| **Code Complexity** | Low-Medium | Data loading + model reuse |
| **Scientific Insight** | ⭐⭐⭐⭐⭐ (Excellent) | Validates multi-day cycling hypothesis |
| **Time to Results** | 1.5-2 days | Need to train 5 models |
| **Reportability** | Very Good | Visual comparison of window sizes |

**What It Does**:
```
Extract windows at 5 different timescales:
  - 6-hour:  captures hourly rhythm baseline
  - 24-hour: current state (single day)
  - 48-hour: ONE bipolar cycle (elevated → depressed)
  - 72-hour: 1.5 cycles (should maximize bipolar signal)
  - 7-day:   weekly stability (unipolar = stable, bipolar = variable)

Train 5 models (reuse existing CNN-LSTM):
  - Each model sees different window size
  - Ensemble: average predictions across 5 models
  
Expected: 48hr model shows ~75%, 72hr shows ~70%
Unipolar doesn't have 48hr cycling → stays at 55-60%
```

**Implementation Plan**:
```python
# 1. Modify data_loader.py: support variable window_minutes parameter
# 2. Create train_exp2_multiscale.py:
#    - Loop over [360, 1440, 2880, 4320, 10080]
#    - Train model for each window size
#    - Evaluate on test set
#    - Ensemble predictions
# 3. Visualize: accuracy vs window_size plot
# 4. Time: 4-6 hours + 2-3 hours training
```

**Success Criteria**:
- At least one window size (48hr or 72hr) ≥ 70% test accuracy
- Ensemble improves over single 24hr model
- Clear plot showing 48hr/72hr > 24hr performance

**If It Fails**:
- Suggests cycling signal not present at any timescale
- Indicates problem is truly data limitation (n=8)
- Pivot to 1B (feature engineering) for alternative approach

---

### 🥉 **RANK 3: Feature Engineering + XGBoost (1B)**

**Why Third**: Most comprehensive signal extraction, good for interpretability

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Feasibility** | ⭐⭐⭐ (Medium) | 6-8 hours coding |
| **Likelihood of Success** | ⭐⭐⭐⭐ (75% confidence) | Prior work showed F1=0.82 with features |
| **Code Complexity** | Medium | Feature extraction + XGBoost |
| **Scientific Insight** | ⭐⭐⭐⭐⭐ (Excellent) | SHAP analysis shows which features matter |
| **Time to Results** | 1.5 days | Feature + training fast |
| **Reportability** | Outstanding | "These 3 features discriminate bipolar" |

**What It Does**:
```
Extract 20+ hand-crafted features:
  - Activity statistics: mean, std, max, min, IQR
  - Temporal dynamics: day-to-day changes, trends
  - Sleep fragmentation: zero-activity periods, consistency
  - Cyclicity: autocorrelation at 24hr/48hr lags, entropy
  
Train XGBoost classifier (shallow trees, 6-10 depth)
Use SHAP for feature importance
Expected: 65-75% accuracy + clear feature interpretation
```

**Implementation Plan**:
```python
# 1. Create extract_features.py: compute 20+ features per window
# 2. Create train_exp2_xgboost.py:
#    - Extract features for all data
#    - Train XGBoost with GridSearch
#    - SHAP analysis
# 3. Visualize: SHAP values, feature importance
# 4. Time: 6-8 hours
```

**Success Criteria**:
- Test accuracy ≥ 65%
- Top 3 features clearly identified by SHAP
- F1-score ≥ 0.60

**If It Fails**:
- Suggests no statistical separability
- Pivot to 3A (statistical tests) to prove signal doesn't exist

---

## Decision Tree: Which to Start With?

```
Do you have 4-6 hours uninterrupted time today?
├─ YES → Start with 1C (Participant Aggregation)
│        Can finish by EOD, get results in 1 day
│        If works: great! If not: pivot to 1A by tomorrow
│
└─ NO → Start with 1B (Feature Engineering)
       More modular, can work in chunks
       Or start with 1A if you prefer visual/experimental approach
```

**My Recommendation**: 
- **TODAY**: Do 1C (Participant Aggregation) → 4 hours → learn if variability matters
- **TOMORROW**: Do 1A (Multi-Scale) if 1C doesn't work → 6 hours → train 5 models overnight
- **DAY 3**: Do 1B if both fail → 8 hours → extract comprehensive features
- **DAY 4**: Statistical analysis (3A) + visualization (3B) → frame findings

---

## Expected Timeline & Milestones

| Date | Approach | Expected Status | Decision Point |
|------|----------|-----------------|----------------|
| **Apr 17 EOD** | 1C | Participant-level LOOCV complete | Does variability separate bipolar? |
| **Apr 18 EOD** | 1A | 5 models trained, ensemble tested | Does 48hr/72hr window show signal? |
| **Apr 19 EOD** | 1B or 3A | XGBoost SHAP + statistical tests | Which features/signal matter? |
| **Apr 20 AM** | Analysis + reporting | Synthesize best findings | What's the story for presentation? |
| **Apr 21 EOD** | Final deliverables | Video + poster ready | **DONE** |

---

## Common Failure Modes & Recovery

| What Goes Wrong | What To Do |
|-----------------|-----------|
| 1C shows no variability difference (p > 0.05) | Don't abandon! This is valuable finding. Move to 1A or 1B to see if longer windows/richer features help. If all fail → honest framing: "Signal too weak at n=8" |
| 1A shows 48hr no better than 24hr | Suggests cycling hypothesis wrong. Move to 1B (maybe activity pattern, not cycling, matters). Or do 3A statistical tests. |
| 1B XGBoost stuck at 65% despite good features | Likely hitting n=8 ceiling. Do 3A to validate statistically. Then frame as: "Features show promise but need larger cohort." |
| Everything fails catastrophically | Do 3A (statistical tests) + 3B (visualizations). Show: "Signal exists statistically (p<0.05) but weak (Cohen's d<0.5), explaining why deep learning fails." Publishable result! |

---

## Code Files Needed (For Reference)

| File | Status | Purpose |
|------|--------|---------|
| `APPROACH_3_NEW_STRATEGIES.md` | ✅ Created | Detailed documentation |
| `classify_by_variability.py` | ⏳ To create | 1C implementation |
| `train_exp2_multiscale.py` | ⏳ To create | 1A implementation |
| `extract_features.py` | ⏳ To create | 1B feature extraction |
| `train_exp2_xgboost.py` | ⏳ To create | 1B XGBoost training |
| `statistical_tests.py` | ⏳ To create | 3A implementation |
| `visualize_signal.py` | ⏳ To create | 3B visualization |

---

## Key Takeaway

**The fundamental question**: Is the bipolar/unipolar distinction visible in 8 bipolar subjects?

- **1C answers**: Does multi-day variability show it?
- **1A answers**: Do multi-day windows show it?
- **1B answers**: Do statistical features show it?
- **3A answers**: Does statistical test confirm it exists?

Pick any approach, run it, get an answer, report honestly. Honesty about sample size limitations is more valuable than false positives.

**Confidence**: ⭐⭐⭐⭐ (80% that at least one approach improves results to 65-70% accuracy)

