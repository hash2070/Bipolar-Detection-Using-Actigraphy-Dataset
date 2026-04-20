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

Our study included 55 participants: 32 healthy controls and 23 patients with mood disorders. Among the mood patients, we focused on 8 with bipolar II depression and 15 with unipolar depression for our main analysis. Each participant wore a wrist activity tracker that recorded minute-by-minute movement data, giving us 1,440 measurements per day (24 hours). The main challenge was the class imbalance—we had almost twice as many unipolar patients as bipolar patients (15 vs. 8).

We tested this problem in two ways. First, we created a baseline to see if we could separate healthy people from depressed people (using all 55 participants). Second, we tackled the harder problem: separating bipolar from unipolar depression (using only the 23 depressed patients).

We then tested three different approaches. Approach 1 used traditional machine learning with different time windows (24, 48, and 72 hours) and simple models like XGBoost and Logistic Regression. Approach 2 used more complex neural networks like LSTM variants and ensemble methods to see if complexity would help. Approach 3 used statistical tests to check if bipolar and unipolar patients showed different activity patterns.

---

## SECTION 3: Key Finding (Center, large figure space)

**Headline (34pt):**
What We Found: An Unexpected Truth

**Visual (use figP_arch_all_comparison.png or fig1_accuracy_comparison.png):**
[Accuracy comparison bar chart showing all approaches]

**Content (21pt body):**

**Best Results:**

| **Approach** | **Accuracy** | **Method** | **What This Means** |
|---|---|---|---|
| Baseline | 88.9% | CNN-LSTM + SMOTE | The model got lucky on our tiny test set—it memorized rather than learned. |
| Approach 1 | 60.9% | Logistic Regression | A simple 5-feature model performed best, beating complex approaches. |
| Approach 2 | ~63% | Advanced networks | Adding more complexity to neural networks gave almost no improvement. |
| Approach 3 | Statistical | Multiple tests | No meaningful difference found between bipolar and unipolar activity patterns. |

**Statistical Testing:** p-value = 0.893 (no significant difference), Cohen's d = -0.060 (negligible effect size).

**The Truth:** When bipolar people are depressed, their activity patterns are indistinguishable from unipolar depression. The biological signal doesn't exist in wrist activity data alone.

---

## SECTION 4: Three Punchlines (Right column, middle)

**Headline (34pt):**
Three Important Lessons

**Content (21pt body + visual elements):**

**1. Big Numbers Don't Always Matter**
Just because a difference is statistically "significant" doesn't mean it's actually useful in real life. Our effect size was so tiny (Cohen's d = -0.06) that a doctor couldn't use it to make real decisions.

**2. Timing Matters**
Looking at activity in 48-hour chunks worked best. When we looked at only 24 hours, the model got confused and memorized patterns that weren't real. When we looked at 72 hours, we missed important details.

**3. Keep It Simple**
A simple model using just 5 activity features (60.9% accuracy) beat fancy neural networks with thousands of settings. Too much complexity on small datasets is like using a sledgehammer to hang a picture.

---

## SECTION 5: Conclusion & Future Work (Bottom right)

**Headline (34pt):**
What's Next?

**Content (21pt body):**

**Why This Matters:** Activity tracking alone cannot detect bipolar vs. unipolar depression. Negative results guide future research—showing where NOT to focus efforts. Understanding what doesn't work is valuable.

**For Future Work:** Need n > 100 bipolar/unipolar pairs and longitudinal tracking across mood episodes. Sleep data (polysomnography) may be more distinctive than activity alone. Multi-modal biomarkers (activity + heart rate + sleep + mood) may work better than single measures.

**Clinical Insight:** Wearables are better for monitoring treatment response than diagnosis. This is a necessary stepping stone in the field.

---

## SECTION 6: References (Bottom left, small 12pt)

**For poster (show subset):**
1. Garcia-Ceja et al. (2018). Depresjon dataset. ACM Multimedia Systems.
2. Fawaz et al. (2019). Deep learning for time series classification. DMKD.
3. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling. JAIR.
4. Cohen (1988). Statistical Power Analysis for Effect Sizes.
5. Kaplan et al. (2007). Actigraphy in psychiatry. J Psychiatric Research.
6. Goodwin & Jamison (2007). Manic-Depressive Illness. Oxford Press.

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
