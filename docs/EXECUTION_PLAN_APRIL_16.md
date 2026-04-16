# EXECUTION PLAN: April 16 → April 17 Presentation
**Status**: LIVE EXECUTION  
**Deadline**: April 17, 2026 (Tomorrow Morning)  
**Execution Mode**: Option A (Local GPU)  
**Time Budget**: 4-5 hours total  

---

## FOLDER STRUCTURE

```
project_root/
├── findings/              # Detailed findings per approach
│   ├── 1C/               # Participant aggregation
│   ├── 1A/               # Multi-scale windows
│   ├── 1B/               # Feature engineering
│   ├── 3A/               # Statistical tests
│   └── 3B/               # Visualization
└── results/              # Final graphs, outputs, compiled results
```

---

## EXECUTION ORDER & TIME BUDGET

| Phase | Task | Time | Status |
|-------|------|------|--------|
| **1** | **1C: Participant Aggregation** | 15 min | ⏳ STARTING |
| **2** | **1A: Multi-Scale Windows** | 30-40 min | ⏳ NEXT |
| **3** | **1B: Feature Engineering** | 20 min | ⏳ PENDING |
| **4** | **3A: Statistical Tests** | 30 min | ⏳ PENDING |
| **5** | **3B: Visualization** | 90 min | ⏳ PENDING |
| | **TOTAL** | **3-5 hours** | ⏳ IN PROGRESS |

---

## APPROACH DETAILS

### **APPROACH 1C: Participant-Level Aggregation (Logistic Regression)**

**File**: `findings/1C/approach_1c_report.md`  
**Code**: `classify_by_variability.py`  
**Expected Result**: 65-70% LOOCV accuracy  
**Hyperparameters to test**:
- C (regularization): [0.1, 1.0, 10.0]
- class_weight: [None, 'balanced']

**Key metric**: LOOCV accuracy (Leave-One-Out CV on 23 participants)

---

### **APPROACH 1A: Multi-Scale Windows (CNN-LSTM)**

**File**: `findings/1A/approach_1a_report.md`  
**Code**: `train_exp2_multiscale.py`  
**Expected Result**: 70-75% test accuracy, 48hr and 72hr >> 24hr  
**Hyperparameters to test**:
- Window sizes: [24hr (1440), 48hr (2880), 72hr (4320) minutes]
- weight_decay: [1e-4, 1e-3]
- dropout: [0.4, 0.6]

**Key metric**: ROC-AUC on test set for each window size

---

### **APPROACH 1B: Feature Engineering + XGBoost**

**File**: `findings/1B/approach_1b_report.md`  
**Code**: `train_exp2_xgboost.py`  
**Expected Result**: 65-75% LOOCV accuracy  
**Hyperparameters to test**:
- max_depth: [3, 5, 7]
- n_estimators: [50, 100]

**Key metric**: Feature importance ranking + LOOCV accuracy

---

### **APPROACH 3A: Statistical Significance Testing**

**File**: `findings/3A/statistical_analysis.md`  
**Code**: `statistical_tests.py`  
**Expected Result**: T-test p-value, Cohen's d effect size  
**Tests**:
- Bipolar variability vs Unipolar variability (t-test)
- Effect size (Cohen's d)

---

### **APPROACH 3B: Visualization & Final Report**

**File**: `findings/3B/visualization_report.md`  
**Code**: `create_visualizations.py`  
**Outputs**:
- ROC curves (1A window sizes comparison)
- Confusion matrices (all approaches)
- Feature importance plot (1B)
- Statistical significance plot (3A)

**Location**: All graphs saved to `results/`

---

## DOCUMENTATION STRUCTURE

Each approach generates:
1. **Detailed Report** (`findings/{APPROACH}/approach_{APPROACH}_report.md`)
   - What we tested
   - Why we chose those hyperparameters
   - Results obtained
   - Interpretations
   - Impact on research

2. **Code File** (`{APPROACH_CODE}.py`)
   - Self-contained, runnable script
   - Comments explaining logic
   - Results printed to console + saved to findings/

3. **Results** (`results/`)
   - CSV files (predictions, metrics)
   - PNG/PDF plots
   - Summary tables

---

## FINAL DELIVERABLES (April 21)

1. ✅ **Comprehensive Final Report** (`FINAL_RESULTS_COMPILATION.md`)
   - Summary of all 5 approaches
   - Impact on bipolar detection research
   - Recommendations for future work

2. ✅ **Presentation-Ready Visuals** (`results/`)
   - ROC curves showing window size effect
   - Feature importance rankings
   - Statistical significance evidence

3. ✅ **Code-to-understand mapping** (documented in findings/)
   - What code file implements what
   - How to reproduce each result
   - Hyperparameter choices explained

---

## STARTING NOW

Beginning with Approach 1C...
