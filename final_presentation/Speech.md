# PowerPoint Presentation: Bipolar Detection via Actigraphy
## A Deep Learning Study

---

## SLIDE 1: Title & Introduction (20 seconds)

**Title:**
Sequence Modeling and Beyond: Wrist Actigraphy for Bipolar-Unipolar Differentiation

**Subtitle:**
A Deep Learning Study on Mental Health Biomarkers

**Authors:** Harsh Mukesh Sharma, Shikha Masurkar  
**Course:** CSCI 5922 — Neural Networks & Deep Learning, CU Boulder

**Speaker 1 (Introduction):**
"Hi Everyone. Today we're presenting our Deep Learning project on detecting bipolar disorder from wrist actigraphy data. My name is Shikha and my teammate's name is Harsh. Let's hear how we tested whether a simple wearable device can distinguish between two types of depression that clinicians often confuse—bipolar and unipolar depression."

---

## SLIDE 2: The Problem (30 seconds)

**Headline (34pt):**
Why Does Bipolar Disorder Go Misdiagnosed for 9.5 Years?

**Visual:** Icon showing confused doctor, antidepressant pill, manic episode warning

**Key Points:**
- Bipolar depression looks identical to unipolar depression clinically
- Both present as low mood and reduced energy
- Misdiagnosis leads to wrong treatment: antidepressants can worsen bipolar disorder
- Doctors currently rely on patient history alone with no objective test
- Current medical practice relies on subjective patient history
- Activity tracking (actigraphy) offers potential as objective measure

**Speaker 1 (Problem Statement):**
"Bipolar depression is misdiagnosed as unipolar depression in nine out of ten cases. When doctors prescribe antidepressants to bipolar patients thinking they have unipolar depression, those medications can actually trigger manic episodes—making the condition worse. Previous studies using simple statistical summaries like average activity and variance have failed to capture the subtle temporal patterns that might distinguish these conditions. There's no objective test today. We just asked ourselves: can a wrist activity tracker combined with machine learning solve this?"

---

## SLIDE 3: Dataset & Methods (60 seconds)

**Headline (34pt):**
How We Tested This: Four Progressive Approaches

**Visual:** Dataset breakdown pie chart + timeline visualization + approach progression diagram

**Key Points:**
- 55 participants: 32 healthy controls, 23 with mood disorders
- Main analysis: 8 bipolar II vs. 15 unipolar depression patients
- Minute-level wrist activity data (1,440 measurements per 24 hours)
- Participant-level stratified splits (prevent data leakage)
- Per-participant z-score normalization (remove baseline differences)

**Four Approaches Tested:**

1. **Baseline Approach:** CNN-LSTM with SMOTE balancing on full 23-patient dataset

2. **Approach 1 (Smaller Dataset):** Downsampling—randomly reducing unipolar samples to exactly match bipolar count for perfectly balanced training

3. **Approach 2 (Advanced Models):** BiLSTM, Attention LSTM, RNN-LSTM, Ensemble voting on full feature set

4. **Approach 3 (Comprehensive Analysis):**
   - **1A:** Multi-scale CNN-LSTM windows—24hr vs 48hr vs 72hr time windows
   - **1B:** Feature engineering + XGBoost to identify key biomarkers
   - **1C:** Participant-level aggregation using Logistic Regression
   - **3A:** T-tests and Mann-Whitney U tests on activity variability; Cohen's d effect size
   - **3B:** Comprehensive visualizations—ROC curves, confusion matrices, feature importance, statistical significance plots

**Speaker 2 (Dataset & Approach):**
"The Depresjon dataset analyzed 55 participants—32 healthy controls and 23 with mood disorders, including 8 bipolar and 15 unipolar patients wearing wrist activity trackers. Here is the visual representation of the dataset. We tested four progressive approaches. First, our baseline CNN-LSTM model with SMOTE balancing. Second, Approach One tested honest downsampling—using only real data by reducing unipolar samples to exactly match bipolar. Third, more complex neural networks like BiLSTM and Attention. And finally, Approach Three—our most comprehensive analysis with five detailed experiments: multi-scale windows, feature engineering, participant aggregation, statistical testing, and visualizations. This systematic progression let us understand where complexity actually helped. The following diagram shows the difference in the architectural structures for the models we trained and tested."

---

## SLIDE 4: Key Findings & Results (45 seconds)

**Headline (34pt):**
What We Found: An Unexpected Truth

**Visual:** Confusion matrices comparison (3 approaches side-by-side) + Accuracy bar chart

**Results Table:**
| Approach | Accuracy | Method | Notes |
|---|---|---|---|
| Baseline | 88.0% | CNN-LSTM + SMOTE | Illusory — model collapse |
| Approach 1 | 65.4% | Downsampling (1:1) | Balance alone insufficient |
| Approach 2 | ~59.7% | Ensemble (3×) | Best architecture variant |
| Approach 3 | 63.4% | CNN-LSTM 48hr Window | Best reliable result |

**Key Findings:**
- Baseline 88.9% accuracy but overfitting on tiny test set
- Best practical: Simple Logistic Regression (60.9%)
- Neural networks added complexity but no improvement
- Statistical testing: p-value = 0.893 (no significant difference)
- Cohen's d = -0.060 (negligible effect size)

**The Truth:** 
"When bipolar people are depressed, their activity patterns are indistinguishable from unipolar depression. The biological signal doesn't exist in wrist activity data alone."

**Speaker 2 (Results):**
"Our best model achieved 88% accuracy, but here's the catch—it was a model collapse, which means the model predicted everyone as unipolar and never detected a single bipolar case. The best practical result came from Approach 3: a CNN-LSTM with 48-hour windows at 63.4%. Interestingly, a simple five-feature logistic regression still achieved 60.9% nearly matching our most complex networks. Most importantly, statistical testing found no meaningful difference between bipolar and unipolar activity patterns. The p-value of 0.893 was essentially random. This tells us something critical: wrist activity data alone isn't the answer. Here we can see the results of all the experiments we ran. Despite increasing model complexity across all four approaches, accuracy barely budged—staying within a narrow 59 to 65 percent range throughout. Every approach clusters near the 65.2% majority-class baseline shown by the dotted line meaning none of our models meaningfully outperformed simply predicting everyone as unipolar. The green row tells the whole story: our best honest result was 63.4% from a 48-hour CNN-LSTM, and every trustworthy model in this table detected zero bipolar cases. Remarkably, the simple 5-feature Logistic Regression detected 4 out of 8 bipolar patients more than any of our deep learning models achieved.

---

## SLIDE 5: Conclusions & Future Work (30 seconds)

**Headline (34pt):**
What's Next?

**Visual:** Future directions roadmap or icon showing integration of multiple biomarkers

**Three Key Lessons:**
1. **Effect Size Matters:** Even statistically significant differences can be too tiny (Cohen's d = -0.06) to be clinically useful
2. **Timing Matters:** 48-hour windows worked best; 24hr overfits, 72hr underfits
3. **Simplicity Wins:** A 5-feature logistic regression (60.9%) beat complex models with thousands of parameters

**Why This Matters:**
- Understanding what doesn't work is just as important as finding what does
- This negative result guides future research toward better biomarkers
- Activity tracking alone cannot diagnose bipolar from unipolar depression

**Future Directions:**
- Need 100+ patients per group for reliable training
- Integrate sleep data (polysomnography) and heart rate variability
- Multi-modal biomarkers (activity + sleep + heart rate + mood) needed
- Wearables excel at monitoring treatment response over time

**Clinical Implications:**
- Activity tracking can't diagnose but can monitor treatment efficacy
- Should be part of multi-modal approach, not standalone diagnostic tool

**Speaker 1 (Takeaway & Future):**
"Our negative result is actually valuable. It tells researchers where NOT to focus. Activity tracking alone simply cannot diagnose bipolar from unipolar depression—the signal isn't in movement alone. To truly solve this, we need larger patient groups, sleep data, heart rate variability, and mood reports combined. While wearables can't diagnose bipolar disorder, they're excellent for monitoring treatment response—tracking whether medication actually improves a patient's activity over time. This is our stepping stone toward better mental health biomarkers. With this, we would like to conclude our research. Thank You!"

---

## PRESENTATION FLOW SUMMARY

| Slide | Speaker | Duration | Content |
|-------|---------|----------|---------|
| 1 | Speaker 1 | 20 sec | Title, hook, context |
| 2 | Speaker 1 | 30 sec | Clinical problem, motivation |
| 3 | Speaker 2 | 45 sec | Dataset, methods, approaches |
| 4 | Speaker 2 | 45 sec | Results, findings, interpretation |
| 5 | Speaker 1 | 30 sec | Conclusions, impact, future work |
| **TOTAL** | **Both** | **~170 sec (2:50)** | **Fits 3-4 min with pauses** |

---

## SPEAKER NOTES FOR VIDEO RECORDING

- Keep pace steady but not rushed
- Pause between sentences for natural pacing
- Emphasize key numbers: 9.5 years misdiagnosis, 88.9% vs 60.9%, p=0.893
- Use hand gestures to point to visuals
- Maintain eye contact with camera
- Practice timing to fit 3-4 minute window
- Speaker transitions should be smooth ("Now I'll hand it over to...")

---
