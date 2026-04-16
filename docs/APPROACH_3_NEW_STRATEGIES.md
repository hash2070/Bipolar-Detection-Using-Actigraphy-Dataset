# APPROACH 3: New Strategies for Bipolar vs Unipolar Detection
## Comprehensive Analysis & Ranked Recommendations

**Document Date:** April 16, 2026  
**Deadline:** April 21, 2026 (5 days remaining)  
**Problem**: Bipolar vs unipolar detection failure (8 bipolar vs 15 unipolar)  
**Context**: Approaches 1 & 2 failed despite balanced data and 4 architecture variants

---

## Executive Summary

The bipolar vs unipolar distinction has failed across **all attempted approaches** because:

1. **Core Issue**: Only 8 bipolar training subjects = ~3,000 windows from 8 people
2. **Signal Mismatch**: 24-hour windows capture static daily patterns, but bipolar/unipolar distinction is about **temporal dynamics and multi-day cycling**
3. **Fundamental Limitation**: Deep learning (CNN-LSTM) is ill-suited for tiny cohorts; domain knowledge required instead

**New Strategy**: Focus on **temporal resolution**, **feature engineering**, and **participant-level learning** rather than window-level deep learning.

---

## Root Cause Analysis

### Why Did Everything Fail?

**Hypothesis Chain**:
1. ✅ **Not architecture** → Verified by 4 variants (Approach 2)
2. ✅ **Not class imbalance** → Verified by perfect 1:1 downsampling (Approach 1)
3. ❌ **Data too small** → 8 subjects = ~1,000 parameters per subject (severe underfitting risk)
4. ❌ **Window size wrong** → 24hr windows show depression state, NOT the cycling that distinguishes bipolar

### Clinical Reality

**What bipolar and unipolar differ on:**
- **Bipolar**: Multi-day mood cycling (high activity → low activity → high, repeat), phase-shifted sleep
- **Unipolar**: Sustained low mood, monotonic activity reduction, consistent low-sleep pattern
- **Current window**: 24 hours = too short to see cycling pattern

**Example**:
```
Bipolar pattern (multi-day cycle):
Day 1: High activity (elevated mood) → activity_mean = 5000, variance = 2000
Day 2: Low activity (depressed)    → activity_mean = 1000, variance = 100
Day 3: High activity (elevated)    → activity_mean = 4500, variance = 1800
Pattern: CYCLICITY visible over 72hr windows

Unipolar pattern (sustained low):
Day 1: Low activity (depressed)    → activity_mean = 1000, variance = 100
Day 2: Low activity (depressed)    → activity_mean = 900, variance = 90
Day 3: Low activity (depressed)    → activity_mean = 950, variance = 95
Pattern: MONOTONIC, no cycling

24-hour windows hide cycling → model sees only single state per window
```

---

## Ranked Recommendations by Feasibility & Impact

### Tier 1: High Feasibility + High Insight (Try First)

#### **1A. Multi-Scale Temporal Windows [PRIORITY: HIGHEST]**

**Problem Addressed**: Current 24hr windows too short to capture bipolar's multi-day cycling

**Approach**:
```python
# Instead of just 1440-minute windows, extract multiple resolutions:
WINDOW_SIZES = {
    '6hr': 360,       # hourly resolution
    '24hr': 1440,     # daily state
    '48hr': 2880,     # see 1 cycle
    '72hr': 4320,     # see 1.5 cycles
    '7day': 10080,    # weekly pattern
}

# For each participant, create features at each scale:
# - 6hr: captures hourly activity rhythm (base features)
# - 24hr: captures daily state (as current model)
# - 48hr: SHOULD reveal bipolar's 2-day cycling signature
# - 72hr: SHOULD amplify cycling signal
# - 7day: captures longer-term stability vs volatility
```

**Implementation**:
1. Modify `data_loader.py` to support variable window sizes
2. Train 5 separate models (one per window size) **OR** hierarchical model
3. Ensemble predictions: bipolar probability = average across 5 models
4. Expected outcome: 48hr and 72hr windows should show clear separation

**Why This Works**:
- Bipolar = high inter-day variability (state changes between days)
- Unipolar = low inter-day variability (consistent low state)
- 48hr windows directly expose this signal
- No architecture change needed, just preprocessing

**Feasibility**: ✅ High (modify data_loader + reuse existing model)  
**Time Required**: 4-6 hours  
**Code Complexity**: Low  
**Likelihood of Improvement**: ⭐⭐⭐⭐⭐ (90% confidence)

---

#### **1B. Feature-Based Discrimination (Not Raw Windows) [PRIORITY: HIGH]**

**Problem Addressed**: Raw sequences lack explicit signal; need domain-informed features

**Features to Extract per Window**:
```python
features = {
    # Circadian rhythm features
    'activity_mean': mean activity
    'activity_variance': activity std dev (lower in unipolar)
    'activity_max': peak activity (lower in unipolar)
    'activity_iqr': interquartile range (more stable in unipolar)
    
    # Temporal dynamics (what changed from previous day?)
    'day_over_day_change': |activity_today - activity_yesterday|
    'trend': slope of activity over week (bipolar = variable, unipolar = downward)
    'coefficient_of_variation': std/mean across days (high for bipolar)
    
    # Sleep/Activity fragmentation
    'activity_zero_periods': count of hourly periods with ~zero activity
    'longest_zero_period': max consecutive hours of inactivity (unipolar has longer stretches)
    'active_hour_consistency': std dev of activity magnitude in active hours
    
    # Cyclicity detection
    'autocorrelation_lag_1day': does activity repeat every 24hr? (lower for bipolar)
    'autocorrelation_lag_2day': does activity repeat every 48hr? (HIGH for bipolar)
    'entropy': Shannon entropy of activity distribution
}
```

**Implementation**:
1. Create `extract_features.py`: compute above features for each 24hr window
2. Use prior day(s) context: features comparing day N to day N-1, N-2, etc.
3. Feed features into shallow classifier:
   - Logistic regression (baseline)
   - XGBoost (handles interactions)
   - Random Forest (feature importance interpretable)
4. Ablation: which features matter most?

**Why This Works**:
- Prior work on Depresjon dataset achieved F1=0.82 with hand-crafted features
- Raw sequences failed (F1=0.52) → engineered features are key
- Bipolar/unipolar differ in **temporal dynamics**, which need explicit measurement

**Feasibility**: ✅ High  
**Time Required**: 6-8 hours  
**Code Complexity**: Medium  
**Likelihood of Improvement**: ⭐⭐⭐⭐ (75% confidence)

---

#### **1C. Participant-Level Aggregation [PRIORITY: HIGH]**

**Problem Addressed**: Current approach: window-level prediction → test collapse. New approach: participant-level statistics

**Idea**:
```python
# Instead of predicting per-window, predict per-participant:

For each participant:
  # Compute 10-14 day aggregate statistics
  aggregate_features = {
    'mean_activity': average daily activity
    'activity_variability_across_days': std of daily means
    'activity_max_across_days': max daily activity observed
    'activity_min_across_days': min daily activity observed
    'range': max - min (bipolar >> unipolar)
    'coefficient_of_variation': variability / mean
    'madrs_start': depression score at start
    'madrs_end': depression score at end
    'depression_change': madrs_end - madrs_start
    'n_days_monitored': length of observation
  }
  
  # Train classifier on these ~10 features from 23 participants
  # Much smaller dataset (23 samples) but more stable signal
```

**Why This Works**:
- Avoids window-level overfitting to participant quirks
- Bipolar show HIGH variability across 10-14 day window
- Unipolar show LOW variability (sustained depression)
- 23 participants = adequate for logistic regression / SVM
- Directly measures the bipolar phenotype (cycling)

**Implementation**:
1. Compute participant-level statistics
2. Train logistic regression: `y = w1*variability + w2*madrs_change + w3*range`
3. Leave-one-participant-out cross-validation (LOOCV)
4. Expect: better generalization, clearer interpretability

**Feasibility**: ✅ High  
**Time Required**: 3-4 hours  
**Code Complexity**: Low  
**Likelihood of Improvement**: ⭐⭐⭐⭐ (70% confidence)

---

### Tier 2: Medium Feasibility + High Insight (If Time Permits)

#### **2A. Temporal Dynamics Modeling: Explicit State Changes**

**Problem Addressed**: Model should predict transitions (depressed→elevated), not just states

**Approach**:
```python
# Current: P(bipolar | window_t)
# New: P(bipolar | transition[t-1, t] + transition[t, t+1])

# Create transition features:
transitions = {
    'activity_change_day1_to_day2': activity[day2] - activity[day1]
    'activity_change_day2_to_day3': activity[day3] - activity[day2]
    'activity_direction_changes': count sign changes in multi-day sequence
    'volatility_index': sum of squared changes across week
}

# These directly measure mood cycling
```

**Rationale**: Bipolar is defined by **state changes**, unipolar by **state stability**

**Feasibility**: ⚠️ Medium (need careful sliding window logic)  
**Time Required**: 6-8 hours  
**Likelihood of Improvement**: ⭐⭐⭐⭐ (65% confidence)

---

#### **2B. Transfer Learning from Healthy vs Depressed [Exp 1]**

**Problem Addressed**: Exp 1 model learned depression signal; can we transfer it?

**Approach**:
```python
# Train base model on Exp 1: healthy vs depressed (32 + 23 = 55 total)
# Model learns: what depression looks like

# Then fine-tune on Exp 2: bipolar vs unipolar (just 23 depressed)
# Model learns: what distinguishes bipolar from unipolar (given both depressed)

# Transfer learning removes need to learn "depressed state" from scratch
# Bipolar-specific signal emerges in fine-tuning layer
```

**Why This Might Work**:
- Exp 1 has 55 participants vs Exp 2's 23
- Learning depression signal first (easy, n=55) → then mood type (hard, n=23)
- Common pattern in transfer learning

**Feasibility**: ⚠️ Medium  
**Time Required**: 4-5 hours  
**Likelihood of Improvement**: ⭐⭐⭐ (50% confidence)

---

#### **2C. Temporal Regularization: Enforce Temporal Consistency**

**Problem Addressed**: Model might be overfitting to noise in individual windows

**Approach**:
```python
# Add regularizer to training: adjacent windows should have similar predictions
# If window[day t] = unipolar, window[day t+1] should probably = unipolar
# (This is clinically realistic: mood state persists day-to-day)

loss_total = loss_classification + λ * loss_temporal_smoothness

loss_temporal_smoothness = Σ |pred[t] - pred[t+1]|²
```

**Why This Works**:
- Prevents erratic predictions (bipolar, unipolar, bipolar, unipolar within same participant)
- Encourages gradual transitions (clinically plausible)
- Helps with class imbalance: can't predict random bipolar

**Feasibility**: ⚠️ Medium (modify loss function)  
**Time Required**: 3-4 hours  
**Likelihood of Improvement**: ⭐⭐⭐ (40% confidence)

---

### Tier 3: High Feasibility + Interpretability Focus (For Report)

#### **3A. Statistical Analysis: T-tests & Feature Importance**

**Problem Addressed**: Do bipolar/unipolar actually differ statistically?

**Approach**:
```python
# Compute statistics for each feature:
bipolar_mean = mean(activity_variability) for bipolar subjects
unipolar_mean = mean(activity_variability) for unipolar subjects

t_stat, p_value = ttest_ind(bipolar_variability, unipolar_variability)

# If p > 0.05: the signal doesn't exist statistically
#   → Deep learning can't find it either
#   → Honest answer: dataset too small to show difference
# If p < 0.05: signal exists but model missed it
#   → Need better features / model
```

**Value**: 
- Clarifies whether failure is data limitation or method limitation
- Publishable finding: "No statistically significant difference found in actigraphy signal between bipolar and unipolar with n=8"

**Feasibility**: ✅ Very High  
**Time Required**: 2-3 hours  
**Likelihood of Adding Insight**: ⭐⭐⭐⭐⭐ (for final report)

---

#### **3B. Visualization: Signal Exploration**

**Problem Addressed**: What DO bipolar/unipolar look like? Maybe current model is right?

**Visualizations**:
```python
# 1. Average activity plot: bipolar vs unipolar over 14 days
#    (overlaid with error bands)

# 2. Inter-day variability box plot: bipolar vs unipolar
#    (show distributions)

# 3. MADRS score vs activity scatter plot
#    (Do depression scores correlate with activity?)

# 4. Individual participant activity over time (14 participants shown)
#    (color by diagnosis, look for visual differences)

# 5. Autocorrelation plots at lag=24hr, 48hr, 72hr
```

**Value**: 
- If visuals show no difference → validates that problem is fundamental, not methodological
- If visuals show clear difference → why didn't model learn it?
- Great for final presentation

**Feasibility**: ✅ High  
**Time Required**: 3-4 hours  
**Likelihood of Insight**: ⭐⭐⭐⭐ (95% for presentation value)

---

## Implementation Roadmap (5 Days Remaining)

### **Day 1-2 (Apr 17-18): Parallel Work on Tier 1**

**Option A**: Multi-Scale Windows (1A)
- Hour 0-2: Modify `data_loader.py` to support multiple window sizes
- Hour 2-4: Create 48hr/72hr/7day datasets
- Hour 4-6: Train 5 models (one per scale)
- Hour 6-8: Evaluate and visualize results

**Option B**: Feature Engineering (1B)
- Hour 0-3: Create `extract_features.py` with 15-20 domain features
- Hour 3-5: Compute features for all data
- Hour 5-8: Train XGBoost + analyze feature importance

**Option C** (Fast): Participant Aggregation (1C)
- Hour 0-1: Compute participant statistics
- Hour 1-2: Train logistic regression + LOOCV
- Hour 2-4: Visualize results
- Hour 4-8: Try other classifiers (SVM, random forest)

**Recommendation**: Start with **1C** (fastest, clearest insight), then **1A** if time.

---

### **Day 3 (Apr 19): Statistical Analysis + Visualization**

- Morning: Statistical t-tests (3A) on raw features
- Afternoon: Comprehensive visualizations (3B)
- Evening: Interpret results, write findings

---

### **Day 4-5 (Apr 20-21): Final Report + Presentation**

- Consolidate results from successful approaches
- Write final report framing findings
- Prepare 4-minute video
- Finish poster for Gather.town

---

## Risk Assessment & Contingency

| Approach | Risk | Contingency |
|----------|------|-------------|
| **1A: Multi-scale windows** | 48hr windows might still not sufficient | Fall back to 1B (features) or 1C (aggregation) |
| **1B: Feature engineering** | Need domain knowledge to select good features | Use automated feature importance (XGBoost SHAP) |
| **1C: Participant aggregation** | Only 23 samples for training | Use LOOCV, logistic regression (robust to small n) |
| **All fail** | Signal genuinely too weak | Pivot to **3A + 3B**: statistical analysis + visualization for honest framing |

---

## Expected Outcomes by Approach

### **Best Case (Multi-Scale + Features + Aggregation)**

```
1A: Multi-scale windows
  - 48hr model: 70-75% accuracy (bipolar/unipolar separated by cycling)
  - 72hr model: 65-70% accuracy
  - Ensemble: 65-70% test ROC-AUC

1B: Feature engineering + XGBoost
  - Test accuracy: 65-75%
  - SHAP analysis shows: "Inter-day variability is key discriminator"
  - F1-score: 0.60-0.70

1C: Participant aggregation
  - LOOCV accuracy: 60-70%
  - Interpretation: variability coefficient predicts bipolar (r=0.5)
  - Clear clinical insight: bipolar = high variability

Combined: At least ONE approach shows 65%+ ROC-AUC
```

### **Realistic Case (One Approach Works)**

- One of 1A/1B/1C shows 60-65% test accuracy
- Others show marginal improvement over baseline
- Statistical analysis (3A) validates signal exists

### **Pessimistic Case (All Fail Again)**

- Outcome: Dataset genuinely too small
- Pivot: Statistical analysis proves no signal at n=8
- Framing: "Contributes to psychiatric literature: 8 bipolar insufficient for actigraphy-based detection"
- Presentation focus: Process (good experimental design) not results

---

## Immediate Next Steps (Action Items)

1. **Decide on Tier 1 approach**: 1C (fastest) vs 1A (most promising) vs 1B (most interpretable)
2. **Create plan**: Which approach to prioritize given deadline
3. **Implement**: Start with selected approach by end of Day 1
4. **Evaluate**: By Day 3 EOD have preliminary results
5. **Pivot if needed**: If approach not working by midday Day 3, switch to alternative

---

## Final Thoughts

**Key insight**: The problem is likely **not neural network architecture**, but rather:
- Signal visibility (need multi-day windows to see cycling)
- Signal representation (need features that measure variability/dynamics)
- Sample size (8 bipolar subjects may be below critical threshold)

**Most Likely Path to Success**: 
1. Validate signal exists statistically (3A)
2. Extract features measuring temporal dynamics (1B)
3. Train shallow classifier on participant-level aggregates (1C)
4. Report honest findings about sample size limitations

**Timeline**: Viable approaches can show results by Apr 19, leaving 2 days for reporting.
