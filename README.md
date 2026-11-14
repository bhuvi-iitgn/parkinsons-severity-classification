# Parkinson's Disease Severity Classification (UCI Telemonitoring)

An end-to-end machine learning pipeline that classifies Parkinson's disease clinical severity (**Mild**, **Moderate**, **Severe**) from non-invasive, sustained-vowel voice recordings, engineered with strict patient-grouped cross-validation to prevent identity leakage.

---

## Overview & Motivation

Monitoring Parkinson's disease progression typically requires frequent in-clinic neurological exams using the **Unified Parkinson's Disease Rating Scale (`total_UPDRS`)**. Because vocal cord tremors and dysphonia appear early and progress alongside motor symptoms, acoustic telemonitoring offers a low-cost way to track disease severity remotely.

This project builds a reproducible, leak-free classification pipeline that predicts a patient's severity tier from acoustic speech biomarkers and basic demographics, achieving an **8x improvement in Severe-stage F1-score** (`0.0400` $\rightarrow$ `0.3236`) across unseen patients.

---

## Dataset & Target Discretization

* **Source:** UCI Parkinson's Telemonitoring Dataset (`data/parkinsons_updrs.data`)
* **Scale:** **5,875** sustained vowel recordings across **42 unique patients** (~140 recordings per patient across 1,030 clinical sitting sessions) with **0 missing values**.
* **Input Features:** **19 raw features** (16 biomedical vocal perturbation/noise measures such as `Jitter`, `Shimmer`, `HNR`, `RPDE`, `DFA`, and `PPE`, plus `age`, `sex`, and `test_time`), expanded to **31 engineered features**.
* **Leakage Prevention:** Identifiers and direct clinical targets (`subject#`, `motor_UPDRS`, `total_UPDRS`) are strictly excluded from the feature matrix.

Continuous `total_UPDRS` scores are discretized into three clinical severity tiers:

| Class | Severity Tier | `total_UPDRS` Range | Recordings | Share | Unique Patients |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **Mild** | $\le 24.0$ | 1,873 | 31.9% | 20 |
| **1** | **Moderate** | $24.0 < x \le 34.0$ | 2,226 | 37.9% | 24 |
| **2** | **Severe** | $> 34.0$ | 1,776 | 30.2% | 19 |

*(Note: Individual patients appear in multiple tiers as their symptoms progress over the 6-month trial.)*

---

## Methodology & Feature Engineering

### 1. Leak-Free Patient-Grouped Validation
A standard random train/test split allows tree models to memorize a patient's unique voice characteristics rather than disease severity. All experiments use **5-fold `StratifiedGroupKFold` grouped strictly by `subject#`**, ensuring **0 overlapping patients** between training and validation folds and evaluating out-of-fold predictions across all **42 unseen patients**.

### 2. Biomedical Feature Engineering (`19` $\rightarrow$ `31` Features)
To capture non-linear vocal degradation and age-adjusted progression, 12 domain features were engineered:
* **Non-Linear Dysphonia Complexity:** `nonlinear_dysphonia = RPDE * DFA * PPE`
* **Amplitude & Frequency Instability:** `jitter_shimmer_prod = Jitter(%) * Shimmer`, `shimmer_apq_ratio`, `jitter_ppq_rap_ratio`
* **Noise-to-Harmonic Ratios:** `ppe_to_hnr = PPE / (HNR + 1e-5)`, `nhr_to_hnr = NHR / (HNR + 1e-5)`
* **Age-Adjusted Biomarkers:** `age_ppe = age * PPE`, `age_dfa = age * DFA`
* **Skewness Normalization:** Log-transformed vocal perturbation metrics (including `log_Jitter(Abs)`).

Five of these engineered features (`age_dfa`, `age_ppe`, `ppe_to_hnr`, `nonlinear_dysphonia`, and `log_Jitter(Abs)`) ranked among the **top 10 most important predictors** in the final model.

---

## Experimental Progression & Hypothesis Testing

### Key Findings
1. **Baseline Severe-Class Collapse (Phase 3):** Default untuned models trained on raw features gravitated toward the majority `Moderate` class, scoring just **0.01** (Logistic Regression) and **0.04** (Random Forest) F1 on `Severe (2)` cases while overfitting to raw patient demographics.
2. **Controlled Ablations (Phase 4B):**
   * *Sex-Normalized Biomarkers (dropping `age`/`sex`):* Dropped Macro F1 to **0.3318**, confirming that `age` carries genuine clinical progression signal rather than pure noise.
   * *Ordinal Regression-to-Bins:* Predicting continuous `total_UPDRS` before binning pulled predictions toward the mean (`Moderate`), collapsing `Severe (2)` F1 to **0.0921**.
   * *Session-Level Probability Averaging (1,030 sittings):* Averaging across recordings within a sitting smoothed out high-confidence `Severe` predictions, lowering Macro F1 to **0.3199**.
3. **Hyperparameter Tuning & Regularization (Phase 4C):** Applying `class_weight='balanced'` alongside depth constraints (`n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`, `max_features=0.3`) prevented patient memorization and maximized macro-averaged generalization.

### 5-Fold Cross-Validation Benchmark (All 42 Unseen Patients)

| Pipeline Stage | Features | Accuracy | Macro F1 | Severe (2) F1 |
| :--- | :---: | :---: | :---: | :---: |
| Phase 3: Baseline Logistic Regression | 19 (Raw) | 0.3810 | 0.3270 | 0.0100 |
| Phase 3: Baseline Random Forest | 19 (Raw) | 0.4084 | 0.3608 | 0.0400 |
| Phase 4A: Regularized Logistic Regression (Balanced) | 31 (Engineered) | 0.3425 | 0.3394 | 0.2355 |
| Phase 4A: Regularized XGBoost (Balanced) | 31 (Engineered) | 0.3435 | 0.3442 | 0.2327 |
| Phase 4A: Regularized Random Forest (Balanced) | 31 (Engineered) | 0.3877 | 0.3837 | 0.2731 |
| **Phase 4C: Tuned Final Random Forest (Production)** | **31 (Engineered)** | **0.4080** | **0.4059** | **0.3236** |

---

## Repository Structure

```text
parkinsons-severity-classification/
├── data/
│   └── parkinsons_updrs.data          # Raw UCI Telemonitoring dataset (5,875 recordings)
├── models/
│   └── parkinsons_rf.joblib           # Serialized final Random Forest model & feature metadata
├── notebooks/
│   └── parkinsons_ml_pipeline.ipynb   # Full EDA, cross-validation, ablations & tuning notebook
├── predict.py                         # Standalone CLI inference script with feature pipeline
├── .gitignore                         # Excludes local virtual environments and cache files
└── README.md                          # Project documentation and benchmarks
```

---

## Quickstart & Inference Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/bhuvi-iitgn/parkinsons-severity-classification.git](https://github.com/bhuvi-iitgn/parkinsons-severity-classification.git)
cd parkinsons-severity-classification
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn joblib
```

### 3. Run Standalone Predictions
Execute `predict.py` to load `models/parkinsons_rf.joblib`, compute the 31 engineered features, and output predicted severity classes and class probabilities:

```bash
python predict.py
```

**Sample Output:**
```text
=== Sample Patient Severity Predictions ===
 Predicted_Class Severity_Label  Prob_Mild  Prob_Moderate  Prob_Severe  Actual_total_UPDRS
               2   Severe (>34)      0.208          0.287        0.505              43.563
               0     Mild (<=24)     0.694          0.137        0.169              12.060
               0     Mild (<=24)     0.799          0.109        0.091              14.109
               0     Mild (<=24)     0.571          0.275        0.154              16.517
               1 Moderate (24-34)    0.005          0.985        0.011              32.000
```

---

## Future Improvements
* **Interactive Web Interface:** Wrap `predict_severity()` in a lightweight Streamlit application (`app.py`) for interactive biomarker sliders and batch CSV uploads.
* **Longitudinal Modeling:** Incorporate patient-specific baseline deltas (`test_time` progression trajectories) to measure how a patient's voice changes relative to their own earlier visits, rather than scoring each recording in isolation.
* **Probability Calibration:** Applying isotonic or Platt scaling to sharpen confidence thresholds for clinical triage alerts, ensuring the predicted percentages closely match true real-world risk.

## Acknowledgments
* **Dataset:** Athanasios Tsanas and Max Little, *UCI Machine Learning Repository - Parkinson's Telemonitoring Dataset*.
* **AI Assistance:** Developed with support from **Google Gemini** for pipeline structuring, editing, coding and Streamlit web deployment.