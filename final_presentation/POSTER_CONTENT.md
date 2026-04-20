# Poster Content - Bipolar Detection via Actigraphy

## Title (55pt, max 2 lines)
**Sequence Modeling and Beyond: Wrist Actigraphy for Bipolar-Unipolar Differentiation**

---

## Authors (top right, small)
Harsh Mukesh Sharma  
CU Boulder, CSCI 5922 Spring 2026  
Course: Neural Networks & Deep Learning

---

## SECTION 1: The Problem (Left column, top)

**Headline (34pt):**
Why Does Bipolar Disorder Go Misdiagnosed for 9.5 Years?

**Content (21pt body):**

Bipolar depression looks identical to unipolar depression clinically—both present as low mood and reduced energy. Misdiagnosis leads to wrong treatment: antidepressants can worsen bipolar disorder. Doctors currently rely on patient history alone with no objective test.

Prior research used simple statistical summaries (average activity, variance) that miss subtle multi-day temporal patterns. We test whether deep learning and statistical analysis can capture the complex temporal structures that distinguish these conditions.

**Historical Context:**
- Bipolar II depression clinically indistinguishable from unipolar depression
- Antidepressants can trigger manic episodes in bipolar patients
- No reliable objective biomarker exists
- Activity tracking (actigraphy) offers potential as objective measure
- Machine learning has found patterns in other psychiatric conditions via wearables

---

## SECTION 2: Dataset & Methods (Center-left, middle)

**Headline (34pt):**
How We Tested This: Our Approaches

**Content (21pt body):**

We analyzed 55 participants (32 controls, 23 with mood disorders): 8 bipolar II vs. 15 unipolar depression patients using minute-level wrist activity data. We tested three approaches: classical ML with different time windows (24-72 hours), advanced neural networks (BiLSTM, Attention LSTM), and statistical hypothesis testing.

---

## SECTION 3: Key Finding (Center, large figure space)

**Headline (34pt):**
What We Found: An Unexpected Truth

**Visual (use figP_arch_all_comparison.png or fig1_accuracy_comparison.png):**
[Accuracy comparison bar chart showing all approaches]

**Content (21pt body):**

| **Approach** | **Accuracy** | **Method** | **Notes** |
|---|---|---|---|
| Baseline | 88.9% | CNN-LSTM + SMOTE | Likely overfitting |
| Approach 1 | 60.9% | Logistic Regression | Best practical result |
| Approach 2 | 63% | Advanced networks | Minimal improvement |
| Approach 3 | 77.9% | Multi-Scale Windows | Best statistical approach |

**Conclusion:** No meaningful difference found between bipolar and unipolar activity patterns. Wrist activity data alone is insufficient for diagnosis.

---

## SECTION 4: Three Punchlines (Right column, middle)

**Headline (34pt):**
Three Important Lessons

**Content (21pt body + visual elements):**

**1. Effect Size Matters:** Even statistically significant differences can be too tiny (Cohen's d = -0.06) to be clinically useful.
**2. Timing Matters:** 48-hour windows worked best; 24hr overfits, 72hr underfits.
**3. Simplicity Wins:** A 5-feature logistic regression (60.9%) beat complex models with thousands of parameters.

---

## SECTION 5: Conclusion & Future Work (Bottom right)

**Headline (34pt):**
What's Next?

**Content (21pt body):**

Wrist activity data alone cannot distinguish bipolar from unipolar depression. To improve this work, we need 100+ patients per group, longitudinal tracking across mood episodes, and multi-modal biomarkers (activity + sleep + heart rate). This negative result is valuable—it guides researchers toward better biomarkers.

---

## SECTION 6: References (Bottom left, small 12pt)

**For poster (show subset):**
1. Garcia-Ceja et al. (2018). Depresjon dataset. ACM Multimedia Systems. doi.org/10.1145/3270459
2. Fawaz et al. (2019). Deep learning for time series classification. Data Mining and Knowledge Discovery.
3. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling. Journal of Artificial Intelligence Research.
4. Cohen (1988). Statistical Power Analysis for Effect Sizes. Lawrence Erlbaum Associates.
5. Kaplan et al. (2007). Actigraphy in psychiatry. Current Psychiatry Reports.
6. Goodwin & Jamison (2007). Manic-Depressive Illness. Oxford University Press.

**Full bibliography:** See `docs/BIBLIOGRAPHY.md` (47 references, all experiments covered)

---

## VISUAL ELEMENTS TO INCLUDE

| Position | Figure | File |
|----------|--------|------|
| **Top left** | Dataset overview | `figA_dataset_overview.png` |
| **Center** | Accuracy comparison | `fig1_accuracy_comparison.png` |
| **Center-right** | Statistical significance | `fig5_statistical_significance.png` |
| **Right** | Model architectures | `figP_arch_all_comparison.png` |
| **Bottom center** | Key insights summary | `figI_three_punchlines.png` |
| **Bottom right** | QR code | (link to video/slides) |

---

## DESIGN NOTES

- **Color scheme:** Red=Bipolar, Blue=Unipolar, Gray=Neutral
- **Alignment:** 3 columns (left/center/right), break into 6 logical zones
- **Typography:** 55pt title, 34pt headers, 21pt body, 12pt refs
- **White space:** Generous margins; align images to yellow grid guidelines
- **Key message:** "Sometimes, null results reveal truth" — frame as strength, not failure

---

## POSTER DIMENSIONS

Standard academic poster: **36" × 48"** (or 48" × 36" landscape)
- Fits Gather.town virtual poster board
- High resolution: 300 DPI for print

---
