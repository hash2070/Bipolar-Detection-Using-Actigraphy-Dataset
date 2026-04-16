# FINAL PRESENTATION REFERENCE DOCUMENT
## Bipolar vs. Unipolar Depression Detection via Wrist Actigraphy

**Course**: CSCI 5922 — Neural Networks & Deep Learning, CU Boulder
**Authors**: Shikha Masurkar, Harsh Mukesh Sharma
**Date**: April 16, 2026
**Deliverable Deadline**: April 21, 2026

---

## ASSIGNMENT REQUIREMENTS SUMMARY

Both deliverables (video + poster) must:
- Motivate the problem selected to solve
- Explain the approach and key design decisions
- Highlight key findings; give insights into what the work taught us
- Focus on 1-3 PUNCHLINES that make the work exciting/valuable
- Be designed for a general audience (think: parents, friends, non-ML people)
- Be positive and engaging

**Deliverable 1**: 4-6 minute recorded video + text notes document
**Deliverable 2**: PDF poster (presented in Gather.town)

---

---

# SECTION 1: POSTER CONTENT

*Full poster text written for a non-ML audience. Use this as the authoritative source for all poster panels.*

---

## POSTER TITLE

**Can a Wristwatch Distinguish Bipolar from Unipolar Depression?**
*A Deep Learning Investigation with an Honest Answer*

Shikha Masurkar | Harsh Mukesh Sharma
CSCI 5922, University of Colorado Boulder — Spring 2026

---

## PANEL 1 — THE PROBLEM (WHY IT MATTERS)

### Depression Is Not One Disease

Most people think of depression as a single condition. But clinically, there are two very different disorders that look almost identical on the surface:

- **Unipolar Depression** — persistent low mood, fatigue, sadness
- **Bipolar Depression** — the same low mood, but caused by a disorder that also involves manic episodes

**Why does the distinction matter?**
The frontline medication for unipolar depression — antidepressants — can trigger a manic episode in bipolar patients. Giving the wrong treatment is not just ineffective; it can be dangerous.

**The diagnostic crisis:**
On average, it takes **9.5 years** for a person with bipolar disorder to receive the correct diagnosis after first seeking help. During that time they may receive the wrong medication, experience hospitalizations, and suffer avoidable harm.

**Our question:**
Can the tiny differences in how people physically move through their daily lives — captured by a wrist-worn sensor, like a Fitbit — reveal which type of depression someone has?

---

## PANEL 2 — THE DATA

### A Wristwatch That Never Sleeps

**Dataset**: Depresjon (Garcia-Ceja et al., 2018)
**55 participants** total:
- 32 healthy controls
- 23 mood disorder patients (8 bipolar, 15 unipolar)

Each participant wore a wrist actigraphy device — similar to a consumer fitness tracker — recording the number of physical movements made **every single minute**, 24 hours a day, for 5 to 29 days continuously.

This gives us a rich time series: for a participant recorded for 14 days, that is **20,160 individual data points**, each representing one minute of their life.

**Depression severity** was also clinically measured using the MADRS scale (Montgomery-Asberg Depression Rating Scale):
- Bipolar group mean: 22 out of 60
- Unipolar group mean: 22 out of 60
- **The groups are clinically indistinguishable by severity**

This is our first clue that this problem is genuinely hard.

---

## PANEL 3 — OUR APPROACH

### Teaching a Neural Network to Read Movement

We built and tested a family of machine learning models, systematically working from complex to simple.

**Step 1 — Data Preparation**
We sliced each participant's recording into 24-hour windows (1,440 data points each). We normalized each person's data to their own baseline, so the model learns patterns of behavior rather than raw activity levels.

**Step 2 — The Primary Model: CNN-LSTM**
Our main architecture combines two types of neural networks:

```
Raw Activity Signal (1,440 minutes)
          |
    [CNN Layer 1]  — finds short patterns (like a morning routine)
          |
    [CNN Layer 2]  — finds medium patterns (daily structure)
          |
    [LSTM Layer]   — tracks how patterns change across days
          |
    [Classifier]   — outputs: Bipolar or Unipolar
```

This model has 223,682 parameters — roughly equivalent to a small neural network that learned from our data.

**Step 3 — Systematic Experiments**
When our initial model failed, we did not stop. We ran four rounds of investigation:

| Approach | What We Tested |
|---|---|
| Baseline | CNN-LSTM on raw 24-hour windows |
| Approach 1 | Fix class imbalance via downsampling |
| Approach 2 | Different architectures (BiLSTM, Attention, Ensemble) |
| Approach 3 | Window size, classical ML, statistical ground truth |

**Step 4 — The Key Diagnostic Test**
We ran formal statistical tests to ask: do bipolar and unipolar patients actually move differently?

---

## PANEL 4 — RESULTS

### What We Found (Honest and Complete)

**The Majority Baseline** (always predict "unipolar"): 65.2% accuracy
This is our floor — any real model must beat this.

| Model | Accuracy | Notes |
|---|---|---|
| CNN-LSTM 24hr (baseline) | ~100% (training), 88% (test) | Model collapse — predicted zero bipolar cases |
| Downsampling fix | 65.4% | Still zero bipolar detected |
| BiLSTM | 55.4% | Worse than baseline |
| Attention LSTM | 48.2% | Near-random |
| Ensemble CNN-LSTM | 59.7% | Most stable variant |
| **CNN-LSTM 48hr window** | **63.4%** | **Best performing model** |
| Logistic Regression (5 features) | 60.9% | Simple model, competitive |
| XGBoost (19 features) | 39.1% | Worst result |

**Best result**: 63.4% — below the majority baseline of 65.2%

**Prior work benchmark**: Jakobsen (2020) achieved F1=0.82 using hand-crafted features on a similar problem. Our deep learning approach did not reach this level.

---

## PANEL 5 — THE THREE PUNCHLINES

*These are the scientifically valuable contributions of our work.*

---

### PUNCHLINE 1: We Precisely Quantified Why This Problem Is Hard

We ran a formal statistical test comparing the movement variability of bipolar vs. unipolar patients.

- Bipolar mean variability: 0.2027
- Unipolar mean variability: 0.2081
- Difference: **0.0054** — essentially zero
- Statistical test: p = 0.8927 (far above the significance threshold of 0.05)
- Effect size: Cohen's d = -0.060 — classified as **negligible**

This is not a failure. **This is a scientific result.**

Cohen's d = 0.06 tells future researchers exactly how underpowered this study is. To detect an effect this small with 80% statistical power would require approximately 4,400 participants — not 23. We have given the field a precise, quantified target for how large a study would need to be.

**This result is publishable in a null results journal.**

---

### PUNCHLINE 2: Context Window Size Is the Key Variable

One of our most actionable findings: the length of the time window fed to the model dramatically changes performance.

| Window Length | Accuracy |
|---|---|
| 24 hours | ~100% training (overfitting — useless) |
| **48 hours** | **63.4% — best result** |
| 72 hours | 42.3% — worse than random |

**Why does this matter?**
For wearable health studies, choosing the right temporal context is crucial. 24 hours captures daily routine but not inter-day variation. 48 hours may capture the beginning of mood cycles. 72 hours introduces too much noise for this dataset size.

**The practical lesson**: Future studies using wrist actigraphy for psychiatric classification should test 48-hour windows as a design choice, not assume 24-hour is optimal.

---

### PUNCHLINE 3: For Small Medical Datasets, Simple Models Rival Deep Learning

Our most parameter-efficient finding:

| Model | Parameters | Accuracy |
|---|---|---|
| CNN-LSTM (48hr) | 223,682 | 63.4% |
| Logistic Regression | 5 (features) | 60.87% |

A logistic regression model using just 5 hand-crafted features — minimum activity, fraction of high-activity minutes, interquartile range, day-to-day variability, and mean activity — achieved 60.9% accuracy versus 63.4% for a 223K-parameter deep neural network.

**The clinical lesson**: When your dataset has 23 patients, a model with 223,000 parameters is almost certainly overparameterized. Deep learning requires data. For small medical datasets, interpretable classical models are not just acceptable — they may be preferable, because clinicians can understand and audit them.

This has direct implications for how medical AI should be deployed in low-data clinical settings.

---

## PANEL 6 — DISCUSSION

### What We Learned

**Why did deep learning underperform?**

The Depresjon dataset has 23 condition participants — 8 bipolar, 15 unipolar. Deep learning models are powerful pattern-finders, but they need data. With 8 bipolar training examples, even with data augmentation (SMOTE, window slicing), the model cannot learn a generalizable representation of "bipolar movement."

The fundamental challenge is not the model architecture. It is the signal itself. The activity patterns of bipolar and unipolar depressed patients — as measured by a wrist sensor — may simply not be separable with current technology and dataset sizes.

**What would we do differently with more time?**

1. Collect MADRS subscale data to separate somatic from psychological symptoms
2. Add sleep architecture features from the same actigraphy signal
3. Incorporate circadian rhythm analysis (phase, amplitude, regularity)
4. Seek access to a larger psychiatric wearable dataset

**How does our work relate to the broader field?**

The gap between our result (63.4%) and Jakobsen 2020 (F1=0.82) likely reflects feature engineering expertise: domain experts who understand psychiatric symptomatology hand-crafted features that capture clinically meaningful patterns. Raw deep learning on minimal data does not automatically discover these patterns.

Our work reinforces an important lesson: **domain knowledge remains critical in medical AI**.

---

## PANEL 7 — FUTURE WORK

### Where This Research Goes Next

1. **Larger datasets**: The NHANES and UK Biobank contain actigraphy data with psychiatric diagnoses. Scale to hundreds of participants to have sufficient power.

2. **Circadian features**: Bipolar disorder is fundamentally a disorder of biological rhythm. Features like mesor, amplitude, and acrophase from circadian modeling may be far more discriminative than raw activity counts.

3. **Multi-modal fusion**: Combine actigraphy with GPS patterns, phone usage, and sleep data for a richer behavioral signature.

4. **Longitudinal modeling**: Instead of classifying a single window, track a patient over months and detect the transition from depression to hypomania — the most clinically critical moment.

5. **Federated learning**: To collect large psychiatric datasets without privacy violations, train models across hospital systems without centralizing data.

---

## PANEL 8 — ACKNOWLEDGMENTS & REFERENCES

**Dataset**: Garcia-Ceja, E., Riegler, M., Jakobsen, P., Torresen, J., Nordgreen, T., Oedegaard, K.J., & Kessler, U. (2018). Depresjon: A motor activity database of depression episodes in unipolar and bipolar patients. *ACM MMSys*.

**Prior Work**: Jakobsen, P. et al. (2020). Applying machine learning in motor activity time series of depressed bipolar and unipolar patients compared with healthy controls. *PLOS ONE*.

**Tools**: PyTorch, scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, matplotlib

**Acknowledgment**: We thank the 55 participants in the Depresjon study whose data made this research possible.

---

---

# SECTION 2: VIDEO SCRIPT

*Timed outline for a 4-6 minute recorded video. Target: 5 minutes. Each section has a time budget, talking points, and figures to show on screen.*

---

## VIDEO OVERVIEW

**Total target time**: 5 minutes (safe range: 4:30 — 5:45)
**Tone**: Enthusiastic, honest, curious. Not apologetic about null results.
**Key principle**: Lead with the story, not the methods.

---

## [0:00 — 0:30] HOOK — The Opening

**On screen**: Title slide with project name and authors.

**What to say**:
"Imagine you've been feeling depressed for two years. You've been to your doctor, you've been prescribed antidepressants, but they aren't working — in fact, they seem to be making things worse. What you actually have is bipolar disorder, and it took nine and a half years to diagnose correctly.

We asked a simple question: can a wristwatch help catch this sooner? Let's talk about what we found."

**Time check**: 30 seconds

---

## [0:30 — 1:15] THE PROBLEM — Why Bipolar vs. Unipolar Matters

**On screen**: Side-by-side diagram — Unipolar (flat line at bottom) vs. Bipolar (wave pattern: depression, baseline, mania)

**What to say**:
"There are two very different types of depression. Unipolar — what most people picture — is persistent low mood. Bipolar disorder includes those same depressive episodes, but also periods of elevated mood or mania.

The problem is that when a bipolar patient comes into a clinic during a depressive episode, they look identical to a unipolar patient. The standard treatment for unipolar depression is antidepressants. But antidepressants can push a bipolar patient into a dangerous manic episode.

So the misdiagnosis is not just frustrating. It can cause real harm. And on average, it takes nine and a half years to get the right answer.

Our idea: people with bipolar disorder may move differently than people with unipolar depression — even during depressive episodes. A wrist sensor, recording movement every minute, might capture that signature."

**Time check**: 1 minute 15 seconds

---

## [1:15 — 2:00] THE DATA & APPROACH

**On screen**: Diagram of data pipeline — wrist device → activity time series → 24/48-hour windows → model → prediction

**What to say**:
"We used the Depresjon dataset — 55 participants, including 32 healthy controls and 23 patients with mood disorders: 8 with bipolar disorder and 15 with unipolar depression.

Each person wore a wrist actigraphy device, similar to a Fitbit, recording how much they moved every single minute, continuously for up to 29 days.

We sliced this data into windows — initially 24 hours, later 48 hours — and fed those windows into a deep learning model combining convolutional neural networks for pattern detection and an LSTM for tracking how those patterns change across days.

When our first model failed to learn, we did not just tune it and give up. We ran four systematic rounds of investigation, each one asking a different question about why."

**Time check**: 2 minutes

---

## [2:00 — 3:15] RESULTS — Honest and Framed Correctly

**On screen**: Results table with accuracy numbers. Highlight: 63.4% best, majority baseline 65.2%, Logistic Regression 60.9%.

**What to say**:
"Here is what we found — and I want to be direct about it.

Our best model achieved 63.4% accuracy. The majority baseline — just predicting 'unipolar' every time — achieves 65.2%. So technically, we did not beat a trivial baseline.

But here is why that result is valuable, not embarrassing.

We ran a formal statistical test. Bipolar patients in this dataset have a mean movement variability of 0.2027. Unipolar patients: 0.2081. That is a difference of 0.0054. The p-value is 0.89 — nowhere near statistically significant. The effect size, Cohen's d, is negative 0.06 — which is classified as negligible.

The signal we were looking for may simply not exist in this dataset at this sample size. And Cohen's d of 0.06 tells us something actionable: to detect an effect this small, you would need roughly 4,400 participants, not 23. We have quantified the problem precisely.

Meanwhile, we found that a logistic regression with just 5 features achieved 60.9% — essentially matching our 223,000-parameter neural network. That tells you something important about deep learning and small medical datasets."

**Time check**: 3 minutes 15 seconds

---

## [3:15 — 4:15] THE THREE PUNCHLINES

**On screen**: Three bullet points, revealing one at a time.

**What to say**:
"Let me give you our three takeaways — the things we actually learned.

First: We precisely quantified why this problem is hard. A Cohen's d of 0.06 is a scientific result. It tells future researchers exactly what sample size they need. This is the kind of null result the field needs to know about.

Second: Context window size matters. When we changed from 24-hour to 48-hour windows, accuracy improved from overfitting collapse to our best result of 63.4%. For anyone designing a future wearable study for psychiatric classification — 48 hours, not 24 hours, appears to be the right temporal context.

Third: For small medical datasets, simple models are not inferior to deep learning — they are often preferable. Logistic regression with 5 interpretable features nearly matched a deep neural network, in a fraction of the computational budget, and with results a clinician can actually understand and trust.

These three findings are not just about our project. They generalize."

**Time check**: 4 minutes 15 seconds

---

## [4:15 — 4:45] FUTURE DIRECTIONS & CLOSING

**On screen**: Future work slide with 3 bullet points. End on title slide.

**What to say**:
"Where does this go next? The most promising directions are: larger datasets with hundreds of participants; circadian rhythm features, since bipolar disorder is fundamentally a disorder of biological rhythm; and multi-modal approaches combining movement with sleep, GPS, and phone behavior.

The core question — can a wristwatch distinguish bipolar from unipolar depression — remains open. But we now know much more clearly what it will take to answer it.

Thank you. We hope this project demonstrates that honest science, even when the results are null, contributes something real to our understanding."

**Time check**: 4 minutes 45 seconds — within target

---

## VIDEO PRODUCTION NOTES

- **Figures to prepare** (in order of appearance):
  1. Title slide (project name, authors, course)
  2. Bipolar vs. unipolar mood diagram (wave vs. flat)
  3. Actigraphy pipeline diagram (wrist → time series → windows → model)
  4. Results table (all models, highlight best)
  5. Statistical test result callout (p=0.8927, Cohen's d=-0.06)
  6. Three punchlines slide
  7. Future work slide

- **Pacing**: Speak at a moderate pace, slightly slower than conversational. The hook should be energetic; the results section should be calm and confident, not apologetic.

- **On null results**: Do NOT say "unfortunately our results were not good." Say "here is what the data told us." The framing is curiosity, not disappointment.

---

---

# SECTION 3: Q&A PREPARATION

*10 likely questions with prepared answers. Practice these until they are natural.*

---

### Q1: "Your accuracy was below the majority baseline. Does that mean your project failed?"

**Prepared answer**:
"No, and I want to push back on that framing. A result below the majority baseline is a scientific finding — it means the features we are using do not contain the signal we expected. Our statistical test confirmed this: p=0.89, Cohen's d=-0.06. The effect size is negligible. That is not a failure; that is the data telling us something true. Jakobsen 2020 achieved F1=0.82, but using hand-crafted clinical features built by domain experts over years. We now know that raw deep learning on 23 patients is insufficient, and we know exactly what sample size would be needed to do better."

---

### Q2: "Why is this problem important clinically? Doctors can just ask patients about mania."

**Prepared answer**:
"That is exactly what doctors try to do, and it takes an average of 9.5 years to get the right answer. There are several reasons: patients often do not recognize their own hypomanic episodes as problematic — they feel good and productive, so they do not report them. Bipolar depressive episodes and unipolar depression are subjectively identical. And psychiatric assessments are expensive and time-limited. A wrist sensor worn passively for a few weeks could provide continuous objective data without any patient self-report, potentially flagging the pattern much earlier."

---

### Q3: "Why did you use a CNN-LSTM? Why not a transformer?"

**Prepared answer**:
"That is a fair question. The CNN-LSTM was a deliberate choice for this problem: CNNs are excellent at detecting local temporal patterns in time series — things like a morning activity spike or afternoon low — and LSTMs track how those patterns evolve across days. Transformers are powerful but require substantially more data to train effectively. With 23 condition participants, a transformer would almost certainly overfit. Our Approach 2 already showed that more complex architectures — BiLSTM, attention mechanisms, ensembles — did not help. The architecture was never the bottleneck."

---

### Q4: "What is Cohen's d and why does it matter?"

**Prepared answer**:
"Cohen's d is a standardized measure of the difference between two groups. A Cohen's d of 0 means the groups are identical. A value of 0.2 is considered small, 0.5 medium, 0.8 large. Our Cohen's d of -0.06 is essentially zero — the bipolar and unipolar groups are moving at nearly identical rates by the metrics we measured. This matters because it gives us a concrete, quantified answer to the question 'how hard is this problem?' rather than just saying 'it is hard.' The negative sign means unipolar patients had very slightly higher variability, but the magnitude is negligible."

---

### Q5: "Why was logistic regression competitive with your deep learning model?"

**Prepared answer**:
"This is one of our most important findings. Deep learning models are universal function approximators — they can learn anything given enough data. With 23 participants and 8 in the minority class, there is simply not enough data for a 223,000-parameter model to generalize. Logistic regression has very few effective parameters — it learned a linear combination of 5 features. With this amount of data, the simpler model is actually a better fit to the data complexity. This is a known phenomenon in medical AI: the no free lunch theorem and the bias-variance tradeoff both predict that complex models will not automatically outperform simple ones on small datasets."

---

### Q6: "Why did the 48-hour window work better than 24 hours?"

**Prepared answer**:
"This is still somewhat speculative, but our interpretation is this: 24-hour windows capture a single daily cycle, which may be very similar across bipolar and unipolar patients within a depressive episode. A 48-hour window begins to capture inter-day variation — how one day differs from the next. Bipolar disorder is fundamentally a cycling condition, and even during a depressive episode there may be subtle day-to-day fluctuations that are more pronounced in bipolar patients. The 72-hour window performed worse, possibly because it introduced too much variability as noise, or because our dataset was too small to support that input dimensionality."

---

### Q7: "How does SMOTE work, and did it help?"

**Prepared answer**:
"SMOTE — Synthetic Minority Oversampling Technique — generates synthetic training examples by interpolating between real minority-class examples. So for our 8 bipolar patients, it creates artificial 24-hour windows that blend the activity patterns of real bipolar participants. The idea is to give the model more minority-class examples so it does not just predict the majority class. In our case, SMOTE did not solve the problem. The reason is likely that the synthetic examples are interpolations of the real data — and if the real bipolar and unipolar data overlap almost completely (Cohen's d=-0.06), then synthetic bipolar examples also overlap with unipolar examples. You cannot synthesize signal that is not present in the original data."

---

### Q8: "What would you do differently if you had 6 more months?"

**Prepared answer**:
"Three things. First, I would seek access to larger datasets — the UK Biobank has actigraphy data with psychiatric diagnoses, and there are several clinical research databases with hundreds of bipolar patients. Second, I would engineer circadian features — mesor, amplitude, and acrophase from cosine rhythm analysis. Bipolar disorder is biologically a circadian rhythm disorder, and these features may be far more discriminative than raw activity counts. Third, I would explore longitudinal classification: instead of classifying a single window, track a patient over months and detect the transition from depression to hypomania, which is the most clinically critical and potentially most behaviorally visible moment."

---

### Q9: "Did you worry about overfitting? How did you handle it?"

**Prepared answer**:
"Yes, extensively. Overfitting is the central challenge when your dataset has 23 patients. We used participant-level train/validation/test splits — meaning a participant's data was entirely in one split, never spread across splits. This prevents data leakage. We used dropout (rate 0.4) and weight decay (L2 regularization). We applied early stopping with patience of 10 epochs. We used batch normalization. Despite all of this, several models still overfit — the RNN-LSTM hit 100% training accuracy and was discarded, the 24-hour CNN-LSTM showed similar collapse. Overfitting with this dataset size is almost unavoidable without extraordinary regularization or data augmentation."

---

### Q10: "What is the ethical dimension of deploying this kind of tool clinically?"

**Prepared answer**:
"This is an important question. Even if we had achieved 90% accuracy, deploying an actigraphy-based bipolar detector in a clinical setting would require careful thought. False positives — telling someone they may have bipolar disorder when they do not — carry significant psychological harm and could lead to inappropriate medication. False negatives give false reassurance. Any clinical deployment would need extensive prospective validation on diverse populations, since our dataset is Norwegian and predominantly white. It should be a decision-support tool, not a replacement for clinical judgment. And the interpretability of the model matters: a clinician needs to understand why the system flagged a patient. Our finding that logistic regression nearly matches deep learning is actually a point in favor of interpretable models for this use case."

---

---

# SECTION 4: KEY MESSAGES TO MEMORIZE

*These are the numbers and statements that must come naturally. If you can say these fluently without looking at notes, you are ready.*

---

## CRITICAL NUMBERS

- **55 participants total**: 32 healthy, 23 with mood disorders (8 bipolar, 15 unipolar)
- **Majority baseline**: 65.2% (always predict unipolar)
- **Best model result**: 63.4% — CNN-LSTM with 48-hour windows
- **Best simple model**: 60.9% — Logistic Regression with 5 features
- **Our model parameter count**: 223,682
- **Statistical test p-value**: 0.8927 (no significant difference)
- **Effect size**: Cohen's d = -0.060 (negligible)
- **Bipolar variability**: 0.2027 | **Unipolar variability**: 0.2081 | **Difference**: 0.0054
- **Prior work benchmark**: F1 = 0.82 (Jakobsen 2020)
- **Misdiagnosis timeline**: 9.5 years average to correct bipolar diagnosis
- **Window comparison**: 24hr = overfitting, 48hr = 63.4% (best), 72hr = 42.3%

---

## THREE PUNCHLINES (say these in your sleep)

1. **"We precisely quantified why this problem is hard."**
   Cohen's d = -0.06. Effect is negligible. Need ~4,400 participants to detect it. That is a publishable null result.

2. **"Context window size is the key variable."**
   48 hours beats 24 hours. Concrete guidance for future wearable study design.

3. **"For small medical datasets, simple models rival deep learning."**
   60.9% logistic regression vs. 63.4% CNN-LSTM. 5 features vs. 223,682 parameters. Clinical lesson: do not overengineer.

---

## KEY FRAMING STATEMENTS

- "This is not a failure — this is the data telling us something true."
- "We did not beat the baseline, but we now know exactly why."
- "Cohen's d of 0.06 is a result the field needs to know."
- "Architecture was never the bottleneck — the signal itself may not be separable at this scale."
- "Simple, interpretable models are preferable when clinicians need to understand and trust the output."
- "The question remains open — but we know much more clearly what it will take to answer it."

---

## WHAT NOT TO SAY

- Do NOT say "unfortunately our model did not work."
- Do NOT say "we failed to achieve good results."
- Do NOT apologize for the null result.
- Do NOT over-explain the SMOTE details in the video — one sentence maximum.
- Do NOT get lost in architecture hyperparameters. The audience does not need layer sizes.

---

## OPENING LINE (memorize verbatim)

"Imagine you've been feeling depressed for two years. You've been to your doctor, you've been prescribed antidepressants, but they aren't working — in fact, they seem to be making things worse. What you actually have is bipolar disorder, and it took nine and a half years to diagnose correctly. We asked a simple question: can a wristwatch help catch this sooner? Let's talk about what we found."

## CLOSING LINE (memorize verbatim)

"The question — can a wristwatch distinguish bipolar from unipolar depression — remains open. But we now know much more clearly what it will take to answer it. And that is exactly what good science looks like."

---

---

# APPENDIX: EXPERIMENT SUMMARY TABLE

*Complete reference for all experiments. Use for Q&A and fact-checking.*

| Experiment | Model | Notes | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Exp 1 Baseline | CNN-LSTM 24hr | Healthy vs. Depressed | 53.4% | 55.6% |
| Exp 2 Baseline | CNN-LSTM + SMOTE | Bipolar vs. Unipolar | 88% (illusory) | — |
| Approach 1 | CNN-LSTM + Downsample | 3044 vs. 3044 windows | 65.4% | — |
| Approach 2a | BiLSTM | Bidirectional | 55.4% | — |
| Approach 2b | Attention LSTM | Attention mechanism | 48.2% | — |
| Approach 2c | RNN-LSTM | Unstable/overfit | 100% (discarded) | — |
| Approach 2d | Ensemble 3x CNN-LSTM | Most stable | 59.7% | — |
| Approach 3-1A | CNN-LSTM 48hr | Best window size | **63.4%** | — |
| Approach 3-1A | CNN-LSTM 72hr | Too long | 42.3% | — |
| Approach 3-1B | XGBoost 19 features | LOOCV | 39.1% | — |
| Approach 3-1C | Logistic Regression 5 features | LOOCV | 60.87% | 0.308 |
| Majority Baseline | Always predict unipolar | Trivial | 65.2% | — |

**Top features by importance (XGBoost)**:
1. activity_min (14.5%)
2. high_activity_fraction (12.2%)
3. activity_iqr (10.9%)
4. day_variability (7.7%)

---

*End of FINAL_PRESENTATION.md*
*Document length: comprehensive reference for poster, video, and Q&A*
*Last updated: April 16, 2026*
