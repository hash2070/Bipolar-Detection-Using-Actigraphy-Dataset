# Project Structure

## Overview

This project is organized by **research approach**, with clear separation of baseline, experimental approaches, documentation, scripts, models, and results.

---

## Directory Tree

```
findings/
├── baseline/
│   ├── exp1_healthy_vs_depressed/
│   │   ├── results/           → exp1_results.json, y_pred/probs/true.npy
│   │   └── figures/           → exp1_confusion_matrix.png, exp1_roc_curve.png
│   │
│   └── exp2_bipolar_vs_unipolar_baseline/
│       ├── results/           → exp2_results.json, y_pred/probs/true.npy
│       └── figures/           → exp2_confusion_matrix.png, exp2_roc_curve.png
│
├── approach1/
│   └── 1A_balanced/              ← ONLY smaller dataset (balanced CNN-LSTM)
│       └── results/              → exp2_approach1_results.json, predictions
│
├── approach2/                     ← Different model architectures
│   ├── bilstm/                    → BiLSTM variant results + predictions
│   ├── attention/                 → Attention LSTM variant results + predictions
│   ├── rnnlstm/                   → RNN-LSTM variant results + predictions
│   └── ensemble/                  → Ensemble (3× soft voting) results + predictions
│
└── approach3/                     ← Detailed analytical study
    ├── 1A/                        ← Window size analysis (24hr/48hr/72hr CNN-LSTM)
    │   ├── docs/                  → APPROACH_1A_COMPLETE_DOCUMENTATION.md
    │   ├── results/               → results_1a.json, results_1a_extended.json
    │   ├── models/                → best_model_24hr.pt, 48hr.pt, 72hr.pt
    │   └── logs/                  → training logs
    │
    ├── 1B/                        ← XGBoost classical ML (19 engineered features)
    │   ├── docs/                  → APPROACH_1B_COMPLETE_DOCUMENTATION.md
    │   ├── results/               → results_1b.json, results_1b_gridsearch.json
    │   └── data/                  → feature_importance.csv, participant_features.csv
    │
    ├── 1C/                        ← Logistic Regression (5 variability features)
    │   ├── docs/                  → APPROACH_1C_DOCUMENTATION.md
    │   ├── results/               → results_1c.json
    │   └── data/                  → participant_features.csv
    │
    ├── 3A/                        ← Statistical significance testing
    │   ├── docs/                  → APPROACH_3A_COMPLETE_DOCUMENTATION.md
    │   └── results/               → results_3a.json (t-stat, p-value, Cohen's d)
    │
    └── 3B/                        ← Visualization generation
        └── docs/                  → APPROACH_3B_COMPLETE_DOCUMENTATION.md
```

---

## Other Folders

```
docs/                             ← All markdown documentation
├── FINAL_PRESENTATION.md          → 4-min video script + poster content + Q&A
├── FINAL_RESULTS_COMPILATION.md   → Complete results summary (Baseline + 3 Approaches)
├── APPROACH_1_DOCUMENTATION.md
├── APPROACH_2_DOCUMENTATION.md
├── APPROACH_3_CHEAT_SHEET.md
└── [11 other documentation files]

scripts/                           ← All Python training & analysis scripts
├── data_loader.py
├── model.py, model_variants.py
├── train_exp1.py, train_exp2.py
├── train_exp2_bilstm.py, train_exp2_attention.py, etc.
├── train_exp2_xgboost.py, statistical_tests.py
├── create_visualizations.py, create_poster_visualizations.py
├── create_architecture_diagrams.py
└── [13 total Python files]

saved_models/                      ← All trained model weights (.pt files)
├── best_model_exp1.pt
├── best_model_exp2.pt (baseline)
├── best_model_exp2_attention.pt, bilstm.pt, rnnlstm.pt, ensemble_0/1/2.pt
└── best_model_24/48/72hr.pt (1A window sizes)

images/                            ← All PNG figures organized by type
├── baseline/                      → exp1/exp2 confusion + ROC curves
├── approach1/                     → fig1-4, fig6, figE, figH
├── approach2/                     → figC, figD (model variant analysis)
├── approach3/                     → fig5, figF (statistical significance)
├── architectures/                 → figJ-figP (all model diagrams)
├── dataset/                       → figA-figB (dataset overview)
└── poster/                        → fig7, figG, figI (presentation-ready)

results/                           ← Raw output data files
├── baseline_exp1/                 → JSON + NPY data (+ PNG figs copied)
├── baseline_exp2/
├── approach1_balanced/
├── approach2_bilstm/approach2_attention/approach2_rnnlstm/approach2_ensemble/
├── approach3_statistical/
├── approach3_visualizations/      → fig1-fig7 (3B outputs)
└── figures/
    ├── analysis/                  → fig1-fig7
    ├── poster/                    → figA-figI
    └── architectures/             → figJ-figP

depresjon/                         ← Original dataset (unchanged)
└── data/
    ├── scores.csv                 → participant metadata
    ├── condition/                 → 23 mood disorder participant CSVs
    └── control/                   → 32 healthy control participant CSVs
```

---

## Key Files at Root

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `requirements.txt` | Python dependencies |
| `PROJECT_STRUCTURE.md` | This file — folder organization |

---

## Quick Navigation

- **For presentation**: See `docs/FINAL_PRESENTATION.md` + `images/poster/`
- **For results summary**: See `docs/FINAL_RESULTS_COMPILATION.md`
- **For individual approach details**: See `findings/approach*/*/docs/APPROACH_*.md`
- **For raw data**: See `findings/*/results/` or `results/*/`
- **For models**: See `saved_models/`
- **For all figures**: See `images/` (organized by approach) or `results/figures/` (all in one place)
- **For code**: See `scripts/`

---

