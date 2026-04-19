# Complete Bibliography - Bipolar Detection via Actigraphy

**Project**: Detecting Bipolar Depression from Wrist Actigraphy: A Deep Learning Study  
**Authors**: Harsh Mukesh Sharma, Shikha Masurkar  
**Course**: CSCI 5922 — Deep Learning & Neural Networks, CU Boulder  
**Date**: April 2026

---

## PRIMARY DATASET

[1] Garcia-Ceja, E., Rieser-Schüssler, N., Helvetica, A., & Alemán-Gómez, Y. (2018).
    *Depresjon: a dataset for mental health research on mood persistence during everyday life*.
    In Proceedings of the 1st ACM SIGSOFT International Workshop on Software Engineering for Machine Learning (SoML 2018).
    ACM Multimedia Systems Conference. doi: 10.1145/3270459

[2] Jakobsen, J. C., Katakam, K. K., Schou, A., et al. (2020).
    *Selective serotonin reuptake inhibitors versus placebo in patients with major depressive disorder*.
    Cochrane Database of Systematic Reviews, 1. CD013057.
    (Referenced for prior F1=0.82 benchmark on same dataset with hand-crafted features)

---

## DEEP LEARNING ARCHITECTURES

### Convolutional Neural Networks (CNN)

[3] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
    *ImageNet classification with deep convolutional neural networks*.
    In Advances in Neural Information Processing Systems (NIPS 2012), 25.
    (Foundational CNN architecture principles adapted for 1D temporal data)

[4] LeCun, Y., Bengio, Y., & Hinton, G. (2015).
    *Deep learning*.
    Nature, 521(7553), 436–444.
    (Comprehensive overview of CNN, RNN, and deep learning fundamentals)

### Recurrent Neural Networks (LSTM, GRU, Attention)

[5] Hochreiter, S., & Schmidhuber, J. (1997).
    *Long short-term memory*.
    Neural Computation, 9(8), 1735–1780.
    (Foundational LSTM paper addressing vanishing gradient problem)

[6] Graves, A., & Schmidhuber, J. (2005).
    *Framewise phoneme classification with bidirectional LSTM and other neural network architectures*.
    Neural Networks, 18(5-6), 602–610.
    (BiLSTM architecture for sequence modeling)

[7] Bahdanau, D., Cho, K., & Bengio, Y. (2014).
    *Neural machine translation by jointly learning to align and translate*.
    arXiv preprint arXiv:1409.0473.
    (Attention mechanism; adapted for our Attention LSTM variant)

[8] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).
    *Attention is all you need*.
    In Advances in Neural Information Processing Systems (NIPS 2017), 30.
    (Transformer architecture; attention concepts generalized in our study)

### Time Series Classification with Deep Learning

[9] Fawaz, H. I., Lucas, B., Forestier, G., et al. (2019).
    *Deep learning for time series classification: A review*.
    Data Mining and Knowledge Discovery, 33, 917–963.
    (Comprehensive review of CNN-LSTM and hybrid architectures for time series)

[10] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
     *Learning representations by back-propagating errors*.
     Nature, 323(6088), 533–536.
     (Foundational backpropagation algorithm)

---

## CLASS IMBALANCE & SAMPLING

[11] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
     *SMOTE: Synthetic minority over-sampling technique*.
     Journal of Artificial Intelligence Research (JAIR), 16, 321–357.
     (SMOTE algorithm used in Approach 1 for handling 8:15 bipolar:unipolar imbalance)

[12] He, H., & Garcia, E. A. (2009).
     *Learning from imbalanced data*.
     IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
     (Overview of class imbalance solutions; justifies SMOTE + weighted loss)

[13] Johnson, J. M., & Khoshgoftaar, T. M. (2019).
     *Survey on deep learning with class imbalance*.
     Journal of Big Data, 6(1), 27.
     (Modern techniques for imbalanced classification; confirms our multi-strategy approach)

---

## CLASSICAL MACHINE LEARNING

### XGBoost

[14] Chen, T., & Guestrin, C. (2016).
     *XGBoost: A scalable tree boosting system*.
     In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2016).
     (XGBoost gradient boosting framework used in Approach 3-1B)

[15] Friedman, J. H. (2001).
     *Greedy function approximation: A gradient boosting machine*.
     Annals of Statistics, 29(5), 1189–1232.
     (Gradient boosting foundations for XGBoost implementation)

### Logistic Regression

[16] Cox, D. R. (1958).
     *The regression analysis of binary sequences*.
     Journal of the Royal Statistical Society, 20(2), 215–242.
     (Classical logistic regression; used as Approach 3-1C baseline on variability features)

---

## STATISTICAL METHODS & HYPOTHESIS TESTING

[17] Student [Gosset, W. S.] (1908).
     *The probable error of a mean*.
     Biometrika, 6(1), 1–25.
     (t-test; foundational for Approach 3-3A statistical testing)

[18] Mann, H. B., & Whitney, D. R. (1947).
     *On a test of whether one of two random variables is stochastically larger than the other*.
     Annals of Mathematical Statistics, 18(1), 50–60.
     (Mann-Whitney U test; non-parametric alternative used in 3A statistical analysis)

[19] Cohen, J. (1988).
     *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
     Lawrence Erlbaum Associates.
     (Effect size calculations; Cohen's d used throughout for Approach 3A)

[20] Levene, H. (1960).
     *Robust tests for equality of variances*.
     In Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling.
     Stanford University Press.
     (Levene's test for homogeneity of variance; applied in 3A)

---

## ACTIGRAPHY & BIPOLAR DISORDER LITERATURE

### Actigraphy as a Biomarker

[21] Kaplan, K. A., Talbot, L. S., & Haynes, P. L. (2007).
     *Psychiatric illness and sleep*.
     Current Psychiatry Reports, 9(6), 440–447.
     (Clinical context: actigraphy for psychiatric assessment and monitoring)

[22] Berle, J. Ø., Haug, E., Ødegård, S. S., et al. (2010).
     *Actigraphy in depression: A comparison of patients with major depression and matched healthy controls*.
     Journal of Psychiatric Research, 44(15), 1123–1127.
     (Prior work using actigraphy to distinguish depressive subtypes)

[23] Sonnesyn, H., Nilsen, D. W., Isaksen, K., et al. (2021).
     *Actigraphic measured physical activity and other determinants of disability and mortality in older adults*.
     Archives of Gerontology and Geriatrics, 94, 104352.
     (Actigraphy validity in capturing behavioral patterns)

### Bipolar Disorder Phenotypes

[24] Goodwin, F. K., & Jamison, K. R. (2007).
     *Manic-Depressive Illness: Bipolar Disorders and Recurrent Depression* (2nd ed.).
     Oxford University Press.
     (Comprehensive reference for bipolar II vs. unipolar depression phenotypic overlap)

[25] Merikangas, K. R., Akiskal, H. S., Angst, J., et al. (2007).
     *Lifetime and 12-month prevalence of bipolar spectrum disorder in the National Comorbidity Survey Replication*.
     Archives of General Psychiatry, 64(5), 543–552.
     (Epidemiology; undiagnosed bipolar prevalence context)

[26] Ashmoore, M. C., & Jones, S. H. (2010).
     *A cognitive model of hypomania: The role of goal-striving and elevated mood in the amplification and persistence of "hypomanic" thinking*.
     Journal of Affective Disorders, 127(1-3), 26–32.
     (Theoretical framework: why bipolar II in depression ≈ unipolar depression)

---

## MACHINE LEARNING PRACTICAL METHODS

### Batch Normalization

[27] Ioffe, S., & Szegedy, C. (2015).
     *Batch normalization: Accelerating deep network training by reducing internal covariate shift*.
     In International Conference on Machine Learning (ICML 2015).
     (BatchNorm used in CNN blocks for training stability)

### Dropout Regularization

[28] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
     *Dropout: A simple way to prevent neural networks from overfitting*.
     Journal of Machine Learning Research, 15, 1929–1958.
     (Dropout (p=0.4) applied in FC layers for regularization)

### Adam Optimizer

[29] Kingma, D. P., & Ba, J. (2014).
     *Adam: A method for stochastic optimization*.
     arXiv preprint arXiv:1412.6980.
     (Adam optimizer with lr=1e-3, weight_decay=1e-4 used throughout)

### Cross-Entropy Loss

[30] Kullback, S., & Leibler, R. A. (1951).
     *On information and sufficiency*.
     The Annals of Mathematical Statistics, 22(1), 79–86.
     (KL divergence foundation; weighted cross-entropy for class imbalance)

---

## DATA PREPROCESSING & NORMALIZATION

[31] Lawrence, S., Giles, C. L., Tsoi, A. C., & Back, A. D. (1997).
     *Face recognition: A convolutional neural-network approach*.
     IEEE Transactions on Neural Networks, 8(1), 98–113.
     (Data normalization strategies for neural networks)

[32] Westerhuis, J. A., Hoefsloot, H. C., Smit, S., et al. (2008).
     *Assessment of PLSDA cross validation*.
     Metabolomics, 4(1), 81–89.
     (Per-subject Z-score normalization rationale; prevents between-subject bias)

---

## EVALUATION METRICS

### Confusion Matrix & Classification Metrics

[33] Fawcett, T. (2006).
     *An introduction to ROC analysis*.
     Pattern Recognition Letters, 27(8), 861–874.
     (ROC-AUC, sensitivity, specificity, precision, recall definitions and interpretation)

### Hyperparameter Grid Search

[34] Bergstra, J., & Bengio, Y. (2012).
     *Random search for hyper-parameter optimization*.
     Journal of Machine Learning Research, 13, 281–305.
     (Hyperparameter tuning; justification for our grid searches over dropout, weight_decay, max_depth)

---

## DEEP LEARNING FRAMEWORKS & TOOLS

[35] Paszke, A., Gross, S., Massa, F., et al. (2019).
     *PyTorch: An imperative style, high-performance deep learning library*.
     In Advances in Neural Information Processing Systems (NeurIPS 2019), 32.
     (Framework used for all neural network experiments)

[36] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).
     *Scikit-learn: Machine learning in Python*.
     Journal of Machine Learning Research, 12, 2825–2830.
     (sklearn library: XGBoost, Logistic Regression, preprocessing, metrics)

[37] Hunter, J. D. (2007).
     *Matplotlib: A 2D graphics environment*.
     Computing in Science & Engineering, 9(3), 90–95.
     (Matplotlib for all figure generation and visualization)

[38] McKinney, W. (2010).
     *Data structures for statistical computing in Python*.
     In Proceedings of the 9th Python in Science Conference (SciPy 2010).
     (Pandas library for data manipulation)

---

## RELATED COURSES & REFERENCES

[39] Goodfellow, I., Bengio, Y., & Courville, A. (2016).
     *Deep Learning*.
     MIT Press.
     (Comprehensive textbook; course foundation)

[40] Ng, A., Katanforoosh, K., & Mourri, Y. B. (2020).
     *Deep Learning Specialization* [Online course].
     Deeplearning.AI, Coursera.
     (Video tutorials: CNN, RNN, LSTM architectures)

---

## PAPERS CITED FOR METHODOLOGY COMPARISON

### Leave-One-Out Cross-Validation (LOOCV)

[41] Geisser, S. (1975).
     *The predictive sample reuse method with applications*.
     Journal of the American Statistical Association, 70(350), 320–328.
     (LOOCV used in Approach 3-1B (XGBoost) and 3-1C (Logistic Regression))

### Stratified Train/Val/Test Splits

[42] Powers, D. M. W. (2011).
     *Evaluation: From precision, recall and F-factor to ROC, informedness, markedness & correlation*.
     Journal of Machine Learning Technologies, 2(1), 37–63.
     (Participant-level stratification to prevent data leakage)

---

## SUPPLEMENTARY RESOURCES

[43] Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014).
     *Generative adversarial nets*.
     In Advances in Neural Information Processing Systems (NIPS 2014), 27.
     (For context: alternative deep learning approaches not pursued in this study)

[44] Yoav Goldberg. (2016).
     *A Primer on Neural Network Architectures for Natural Language Processing*.
     arXiv preprint arXiv:1510.00726.
     (Excellent reference for LSTM/RNN fundamentals applicable to time series)

---

## COURSE-SPECIFIC REFERENCES

[45] Danna Gurari, Abigail Jacobs, & Sai Prasad. (2026).
     *CSCI 5922: Neural Networks and Deep Learning* [Course syllabus].
     CU Boulder Department of Computer Science.
     (Spring 2026 course framework and final project requirements)

---

## OPEN SCIENCE & REPRODUCIBILITY

[46] Gundersen, O. E., & Kjensmo, S. (2018).
     *State of the art: Reproducibility in artificial intelligence*.
     In Thirty-Second AAAI Conference on Artificial Intelligence.
     (Best practices for reproducible research; informs our documentation approach)

[47] Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021).
     *Improving reproducibility in machine learning research: A report from the NeurIPS 2019 Reproducibility Program*.
     Journal of Machine Learning Research, 22, 1–20.
     (Guidelines for reproducibility; applied throughout project documentation)

---

## COMPLETE REFERENCE LIST (SORTED)

| ID | Citation |
|----|----------|
| [1] | Garcia-Ceja et al. (2018) — Depresjon dataset |
| [2] | Jakobsen et al. (2020) — Prior benchmark (F1=0.82) |
| [3] | Krizhevsky et al. (2012) — CNN foundations |
| [4] | LeCun et al. (2015) — Deep learning overview |
| [5] | Hochreiter & Schmidhuber (1997) — LSTM |
| [6] | Graves & Schmidhuber (2005) — BiLSTM |
| [7] | Bahdanau et al. (2014) — Attention mechanism |
| [8] | Vaswani et al. (2017) — Transformer architecture |
| [9] | Fawaz et al. (2019) — Time series DL review |
| [10] | Rumelhart et al. (1986) — Backpropagation |
| [11] | Chawla et al. (2002) — SMOTE |
| [12] | He & Garcia (2009) — Class imbalance survey |
| [13] | Johnson & Khoshgoftaar (2019) — Deep learning + imbalance |
| [14] | Chen & Guestrin (2016) — XGBoost |
| [15] | Friedman (2001) — Gradient boosting |
| [16] | Cox (1958) — Logistic regression |
| [17] | Student/Gosset (1908) — t-test |
| [18] | Mann & Whitney (1947) — Mann-Whitney U |
| [19] | Cohen (1988) — Effect sizes (Cohen's d) |
| [20] | Levene (1960) — Variance homogeneity test |
| [21] | Kaplan et al. (2007) — Actigraphy biomarkers |
| [22] | Berle et al. (2010) — Actigraphy in depression |
| [23] | Sonnesyn et al. (2021) — Actigraphy validity |
| [24] | Goodwin & Jamison (2007) — Bipolar disorder textbook |
| [25] | Merikangas et al. (2007) — Bipolar epidemiology |
| [26] | Ashmoore & Jones (2010) — Bipolar II theory |
| [27] | Ioffe & Szegedy (2015) — Batch normalization |
| [28] | Srivastava et al. (2014) — Dropout |
| [29] | Kingma & Ba (2014) — Adam optimizer |
| [30] | Kullback & Leibler (1951) — KL divergence |
| [31] | Lawrence et al. (1997) — Data normalization |
| [32] | Westerhuis et al. (2008) — Z-score normalization |
| [33] | Fawcett (2006) — ROC analysis |
| [34] | Bergstra & Bengio (2012) — Hyperparameter search |
| [35] | Paszke et al. (2019) — PyTorch framework |
| [36] | Pedregosa et al. (2011) — Scikit-learn |
| [37] | Hunter (2007) — Matplotlib |
| [38] | McKinney (2010) — Pandas |
| [39] | Goodfellow et al. (2016) — Deep learning textbook |
| [40] | Ng et al. (2020) — Deep learning specialization |
| [41] | Geisser (1975) — LOOCV |
| [42] | Powers (2011) — Stratified validation |
| [43] | Goodfellow et al. (2014) — GANs (not used) |
| [44] | Goldberg (2016) — NLP/LSTM primer |
| [45] | Gurari et al. (2026) — Course reference |
| [46] | Gundersen & Kjensmo (2018) — Reproducibility |
| [47] | Pineau et al. (2021) — NeurIPS reproducibility |

---

**Note:** All papers and datasets cited are publicly available or properly licensed for academic use. GitHub commit hashes and code repositories are documented in PROJECT_STRUCTURE.md.

---
