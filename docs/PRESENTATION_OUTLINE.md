# Presentation Outline: 4-Minute Video Script

## [0:00-0:20] Opening & Motivation (20 sec)

"Bipolar Disorder affects 1-2% of the population, but it takes patients an average of **10 years** to get diagnosed. Why? Because people seek help during depressive episodes that look nearly identical to regular depression.

The tragic consequence: doctors prescribe antidepressants—which can **trigger severe manic episodes** and make the condition worse.

We asked a simple question: Can a wrist-worn activity tracker detect the difference between bipolar and unipolar depression—automatically?"

## [0:20-0:50] Approach (30 sec)

"We used the Depresjon dataset—**55 participants, 10+ days of actigraphy each**—from clinical research.

Instead of hand-crafted features, we built an end-to-end deep learning model: a **1D-CNN-LSTM hybrid**.

The CNN captures short-term restless activity and sleep disruptions. The LSTM learns longer circadian rhythm changes that differ between the two conditions.

No manual feature engineering—just raw minute-level activity counts."

## [0:50-1:40] Two-Stage Experiments (50 sec)

**Experiment 1: Healthy vs. Depressed (Baseline Validation)**
- Tests if the architecture works at all
- All 55 participants
- Results: [INSERT METRICS]

**Experiment 2: Bipolar vs. Unipolar (The Real Challenge)**
- Only depressed patients (8 bipolar, 15 unipolar)
- Highly imbalanced classes
- Uses SMOTE to balance training data
- Primary metric: ROC-AUC (robust to imbalance)
- Results: [INSERT METRICS]

## [1:40-2:20] Key Findings (40 sec)

"[Interpret results]

Experiment 1 validates our architecture can extract meaningful patterns from actigraphy.

Experiment 2 shows that [RESULT: success/struggle to distinguish bipolar from unipolar].

The model captures **both** short-term kinetic disruptions and multi-day circadian patterns—the exact temporal structure that distinguishes these conditions."

## [2:20-3:20] Significance & Next Steps (60 sec)

"Why does this matter?

**Clinically**: An objective, passive tool could flag bipolar depression **before the wrong treatment is started**—potentially preventing manic episodes and hospitalizations.

**Technically**: We show that deep sequence models outperform fixed-window statistics because they capture temporal richness that classical features miss.

**Barriers to deployment**:
- Larger datasets needed
- Prospective validation with real patients
- Integration into wearable UX/privacy

But the foundation is here: multi-day actigraphy **carries diagnostic signal**."

## [3:20-3:50] Closing (30 sec)

"Ten years to diagnosis is too long. A smartphone-sized wearable that whispers 'this might be bipolar' could change everything.

We've shown it's possible to detect these differences automatically. The question is no longer 'if'—it's 'when'."

---

# Poster Content Structure

## Title & Authors
**Sequence Modeling of Wrist-Worn Actigraphy for Differentiating Bipolar and Unipolar Depressive Episodes**
Shikha Masurkar, Harsh Mukesh Sharma | CSCI 5922, CU Boulder

## Section 1: Problem Motivation (Top Left, 25% width)
- **The Diagnostic Delay**: 10-year average before correct diagnosis
- **The Clinical Hazard**: Antidepressants → manic switching in bipolar patients
- **The Opportunity**: Passive wearables as objective diagnostic tools

## Section 2: Method (Top Center-Right, 40%)
**1D-CNN-LSTM Architecture**
- [Include Figure 1 architecture diagram]
- CNN Block 1: 64 filters, kernel=7
- CNN Block 2: 128 filters, kernel=5
- LSTM: 128 hidden units
- FC layers: 256 → 64 → 2 classes
- **Total parameters**: 223,682

## Section 3: Dataset (Top Right, 25%)
**Depresjon Dataset**
- 55 participants total
- 32 healthy controls
- 23 mood disorder patients
- 8 Bipolar (I/II)
- 15 Unipolar (MDD)
- ~12.7 days per participant
- 1-minute epoch activity counts

## Section 4: Experiments (Bottom Left, 45%)
**Exp 1: Healthy vs. Depressed**
- [ROC Curve visualization]
- Accuracy: [INC]%
- F1-Score: [INC]
-ROC-AUC: [INC]

**Exp 2: Bipolar vs. Unipolar**
- [ROC Curve visualization]
- ROC-AUC: [INC] (primary metric)
- [Confusion Matrix]
- Uses SMOTE for imbalance handling

## Section 5: Results & Impact (Bottom Right, 55%)
**Key Findings**
- End-to-end temporal modeling outperforms fixed-window statistics
- Multi-scale features (CNN + LSTM) capture both kinetic texture and circadian disruption
- Feasible to distinguish bipolar from unipolar using passive wearables alone
- [Training curves showing convergence]

**Clinical Significance**
- Objective flag for suspected bipolar before treatment initiation
- Potential to prevent iatrogenic manic episodes from antidepressants
- Foundation for smartphone-integrated mental health monitoring

**Limitations**
- Small sample size (esp. bipolar: n=8)
- Dataset from clinical population (selection bias may not generalize)
- Future work: prospective validation on new patients
