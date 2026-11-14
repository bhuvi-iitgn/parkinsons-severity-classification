# Parkinson's Disease Severity Classification (UCI Telemonitoring)
Multi-Stage Parkinson’s Disease Severity Classification using Remote Voice Telemonitoring (ES 335)

An end-to-end machine learning pipeline that classifies Parkinson's disease severity into three clinical stages (**Mild**, **Moderate**, **Severe**) from sustained vowel voice recordings without patient identity leakage.

## Project Structure
- `data/parkinsons_updrs.data` — UCI Parkinson's Telemonitoring dataset (5,875 recordings across 42 patients).
- `notebooks/parkinsons_ml_pipeline.ipynb` — Exploratory Data Analysis, 3-class target binning, patient-grouped cross-validation, feature engineering, ablation testing, and hyperparameter tuning.
- `models/parkinsons_rf.joblib` — Final tuned Random Forest classifier artifact.
- `predict.py` — Standalone inference script for scoring new voice recordings.

## Methodology
- **Target Discretization:** Continuous `total_UPDRS` scores are mapped into `Mild (<=24)`, `Moderate (24–34)`, and `Severe (>34)`.
- **Leak-Free Validation:** Uses 5-fold `StratifiedGroupKFold` grouped strictly by `subject#` so recordings from the same patient never overlap between training and validation folds.
- **Domain Feature Engineering:** Expands 19 raw acoustic/demographic variables into 31 features, including non-linear dysphonia complexity (`RPDE * DFA * PPE`), harmonic-to-noise ratios (`ppe_to_hnr`), age-adjusted vocal biomarkers (`age_dfa`, `age_ppe`), and log-transformed perturbation metrics.
- **Final Model:** Class-balanced `RandomForestClassifier` (`n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`, `max_features=0.3`).

## 5-Fold Cross-Validation Progression (All 42 Unseen Patients)
| Stage | Accuracy | Macro F1 | Severe (2) F1 |
| :--- | :---: | :---: | :---: |
| Phase 3: Baseline Random Forest (19 raw features) | 0.4084 | 0.3608 | 0.0400 |
| Phase 4A: Regularized RF (31 engineered features) | 0.3877 | 0.3837 | 0.2731 |
| **Phase 4C: Tuned Final RF (31 engineered features)** | **0.4080** | **0.4059** | **0.3236** |

## Usage
```bash
python predict.py