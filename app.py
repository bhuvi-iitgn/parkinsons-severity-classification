import numpy as np
import pandas as pd
import streamlit as st
from predict import predict_severity

st.set_page_config(
    page_title="Parkinson's Severity Classifier",
    page_icon="🩺",
    layout="wide",
)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv("data/parkinsons_updrs.data")


raw_df = load_reference_data()
feature_medians = raw_df.median(numeric_only=True)

st.title("Parkinson's Disease Severity Classifier")
st.caption(
    "Non-invasive UPDRS severity triage (Mild, Moderate, Severe) from sustained-vowel "
    "voice biomarkers - powered by a patient-grouped Random Forest model."
)

with st.sidebar:
    st.header("Project Information")
    st.markdown(
        """
        **Project:** Parkinson's Disease Severity Classification  
        **Author:** Bhuvi Kalarwal  
        **Roll no.:** 24110076  
        **Course:** ES 335: Machine Learning  
        **Institution:** IIT Gandhinagar

        ---
        **Resources:**  
        - [GitHub Repository](https://github.com/bhuvi-iitgn/parkinsons-severity-classification)  
        - **Dataset:** UCI Parkinson's Telemonitoring (5,875 recordings, 42 patients)  

        ---
        **Acknowledgments:**  
        Built with assistance from **Google Gemini** for pipeline structuring, editing, coding and Streamlit web deployment.
        """
    )

# Update your existing tabs definition near the middle of app.py to add the 4th tab:
tab_single, tab_batch, tab_about, tab_report = st.tabs(
    [
        "Single Recording Predictor",
        "Batch CSV Scoring",
        "Model & Benchmarks",
        "Full Study & Methodology",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: Interactive Single-Patient / Single-Recording Predictor
# ---------------------------------------------------------------------------
with tab_single:
    col_inputs, col_results = st.columns([1.15, 1.0], gap="large")

    with col_inputs:
        st.subheader("Patient & Acoustic Biomarkers")

        if st.button("Load Random Sample Recording from Dataset"):
            sample_row = raw_df.sample(1).iloc[0]
            st.session_state["age"] = int(sample_row["age"])
            st.session_state["sex"] = int(sample_row["sex"])
            st.session_state["ppe"] = float(sample_row["PPE"])
            st.session_state["dfa"] = float(sample_row["DFA"])
            st.session_state["rpde"] = float(sample_row["RPDE"])
            st.session_state["hnr"] = float(sample_row["HNR"])
            st.session_state["jitter"] = float(sample_row["Jitter(%)"])
            st.session_state["shimmer"] = float(sample_row["Shimmer"])
            st.session_state["nhr"] = float(sample_row["NHR"])
            st.session_state["actual_updrs"] = float(sample_row["total_UPDRS"])

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider(
                "Age (years)",
                min_value=35,
                max_value=85,
                value=st.session_state.get("age", int(feature_medians["age"])),
            )
            ppe = st.slider(
                "PPE (Pitch Period Entropy)",
                min_value=0.02,
                max_value=0.75,
                value=st.session_state.get("ppe", round(float(feature_medians["PPE"]), 3)),
                step=0.005,
                format="%.3f",
            )
            dfa = st.slider(
                "DFA (Detrended Fluctuation)",
                min_value=0.50,
                max_value=0.87,
                value=st.session_state.get("dfa", round(float(feature_medians["DFA"]), 3)),
                step=0.005,
                format="%.3f",
            )
            rpde = st.slider(
                "RPDE (Recurrence Entropy)",
                min_value=0.15,
                max_value=0.97,
                value=st.session_state.get("rpde", round(float(feature_medians["RPDE"]), 3)),
                step=0.005,
                format="%.3f",
            )

        with c2:
            sex_label = st.selectbox(
                "Biological Sex",
                options=["Male (0)", "Female (1)"],
                index=st.session_state.get("sex", int(feature_medians["sex"])),
            )
            sex = 0 if sex_label.startswith("Male") else 1

            hnr = st.slider(
                "HNR (Harmonics-to-Noise Ratio)",
                min_value=1.5,
                max_value=38.0,
                value=st.session_state.get("hnr", round(float(feature_medians["HNR"]), 2)),
                step=0.25,
            )
            jitter = st.slider(
                "Jitter (%)",
                min_value=0.0008,
                max_value=0.1000,
                value=st.session_state.get("jitter", round(float(feature_medians["Jitter(%)"]), 4)),
                step=0.0005,
                format="%.4f",
            )
            shimmer = st.slider(
                "Shimmer (Amplitude Perturbation)",
                min_value=0.003,
                max_value=0.270,
                value=st.session_state.get("shimmer", round(float(feature_medians["Shimmer"]), 4)),
                step=0.001,
                format="%.4f",
            )
            nhr = st.slider(
                "NHR (Noise-to-Harmonics Ratio)",
                min_value=0.0002,
                max_value=0.7500,
                value=st.session_state.get("nhr", round(float(feature_medians["NHR"]), 4)),
                step=0.001,
                format="%.4f",
            )

    # Build a complete 19-feature input row anchored to dataset medians
    input_row = feature_medians.to_dict()
    jitter_scale = jitter / max(float(feature_medians["Jitter(%)"]), 1e-6)
    shimmer_scale = shimmer / max(float(feature_medians["Shimmer"]), 1e-6)

    input_row.update(
        {
            "age": age,
            "sex": sex,
            "PPE": ppe,
            "DFA": dfa,
            "RPDE": rpde,
            "HNR": hnr,
            "Jitter(%)": jitter,
            "Jitter(Abs)": float(feature_medians["Jitter(Abs)"]) * jitter_scale,
            "Jitter:RAP": float(feature_medians["Jitter:RAP"]) * jitter_scale,
            "Jitter:PPQ5": float(feature_medians["Jitter:PPQ5"]) * jitter_scale,
            "Jitter:DDP": float(feature_medians["Jitter:DDP"]) * jitter_scale,
            "Shimmer": shimmer,
            "Shimmer(dB)": float(feature_medians["Shimmer(dB)"]) * shimmer_scale,
            "Shimmer:APQ3": float(feature_medians["Shimmer:APQ3"]) * shimmer_scale,
            "Shimmer:APQ5": float(feature_medians["Shimmer:APQ5"]) * shimmer_scale,
            "Shimmer:APQ11": float(feature_medians["Shimmer:APQ11"]) * shimmer_scale,
            "Shimmer:DDA": float(feature_medians["Shimmer:DDA"]) * shimmer_scale,
            "NHR": nhr,
        }
    )

    input_df = pd.DataFrame([input_row])
    pred_df = predict_severity(input_df).iloc[0]

    with col_results:
        st.subheader("Predicted Clinical Severity")

        pred_class = int(pred_df["Predicted_Class"])
        severity_label = pred_df["Severity_Label"]

        if pred_class == 0:
            st.success(f"**Predicted Tier:** {severity_label}")
        elif pred_class == 1:
            st.warning(f"**Predicted Tier:** {severity_label}")
        else:
            st.error(f"**Predicted Tier:** {severity_label}")

        if "actual_updrs" in st.session_state:
            st.caption(
                f"Ground-truth `total_UPDRS` for loaded sample: **{st.session_state['actual_updrs']:.2f}**"
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Mild (<=24)", f"{pred_df['Prob_Mild']:.1%}")
        m2.metric("Moderate (24-34)", f"{pred_df['Prob_Moderate']:.1%}")
        m3.metric("Severe (>34)", f"{pred_df['Prob_Severe']:.1%}")

        prob_chart_df = pd.DataFrame(
            {
                "Severity Tier": ["Mild (<=24)", "Moderate (24-34)", "Severe (>34)"],
                "Probability": [
                    pred_df["Prob_Mild"],
                    pred_df["Prob_Moderate"],
                    pred_df["Prob_Severe"],
                ],
            }
        ).set_index("Severity Tier")

        st.bar_chart(prob_chart_df, width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: Batch CSV Upload & Scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Patient Recording Scoring")
    st.write(
        "Upload a CSV containing the 19 raw acoustic/demographic columns, or run a live demo "
        "on 15 randomly sampled recordings from the UCI dataset."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    use_demo_batch = st.checkbox("Use 15 sample recordings from dataset", value=(uploaded_file is None))

    batch_source = None
    if uploaded_file is not None:
        batch_source = pd.read_csv(uploaded_file)
    elif use_demo_batch:
        batch_source = raw_df.sample(15, random_state=42).reset_index(drop=True)

    if batch_source is not None:
        try:
            scored_df = predict_severity(batch_source)
            if "total_UPDRS" in batch_source.columns:
                scored_df["Actual_total_UPDRS"] = batch_source["total_UPDRS"].values

            st.dataframe(scored_df, width="stretch")

            csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_bytes,
                file_name="parkinsons_severity_predictions.csv",
                mime="text/csv",
                width="stretch",
            )
        except KeyError as exc:
            st.error(f"Missing required raw feature column in uploaded CSV: {exc}")

# ---------------------------------------------------------------------------
# TAB 3: Methodology & Benchmarks
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Pipeline Architecture & Validation Benchmarks")
    st.markdown(
        """
        - **Dataset:** 5,875 sustained vowel voice recordings across **42 unique patients** (UCI Telemonitoring).
        - **Leak-Free Evaluation:** 5-fold `StratifiedGroupKFold` grouped strictly by `subject#` so recordings from the same patient never overlap across train and validation folds.
        - **Feature Engineering:** Expands 19 raw acoustic/demographic inputs into **31 engineered features**, including non-linear dysphonia complexity (`RPDE * DFA * PPE`), harmonic-to-noise ratios (`ppe_to_hnr`), and age-adjusted biomarkers (`age_dfa`, `age_ppe`).
        """
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Final Macro F1 (42 Unseen Patients)", "0.4059", "+0.0451 vs Baseline")
    b2.metric("Severe-Stage (>34) F1", "0.3236", "8.1x vs Baseline (0.0400)")
    b3.metric("Engineered Features", "31", "+12 Domain Biomarkers")

# ---------------------------------------------------------------------------
# TAB 4
# ---------------------------------------------------------------------------
with tab_report:
    st.header("Full Study & Methodology Report")
    st.caption(
        "ES 335: Machine Learning (IIT Gandhinagar) | Author: Bhuvi Kalarwal (Roll No. 24110076)"
    )

    st.subheader("1. Research Question & Problem Formulation")
    st.markdown(
        """
        * **Research Question:** Can sustained-vowel acoustic biomarkers and basic patient demographics classify Parkinson's disease clinical severity across **unseen patients** without requiring an in-clinic motor examination?
        * **Unit of Observation (1 Row):** One sustained-vowel phonation recording (`"aaaah"`) captured via home telemonitoring.
        * **Target Variable & Clinical Tiers:** Continuous `total_UPDRS` (Unified Parkinson's Disease Rating Scale) discretized into three actionable triage tiers:
          * **Mild (`total_UPDRS <= 24`):** Early motor/speech involvement (`36.9%` of recordings).
          * **Moderate (`24 < total_UPDRS <= 34`):** Intermediate progression (`41.4%` of recordings).
          * **Severe (`total_UPDRS > 34`):** Advanced impairment requiring closer clinical follow-up (`21.7%` of recordings).
        * **Prediction-Time Scope:** Uses only non-invasive variables available during remote voice screening (`age`, `sex`, and 16 acoustic features). `motor_UPDRS` (a direct sub-score of `total_UPDRS`), `subject#`, and `test_time` are strictly excluded to prevent target leakage and patient-identity memorization.
        """
    )

    st.subheader("2. Dataset & Data Understanding")
    st.markdown(
        """
        * **Source:** UCI Parkinson's Telemonitoring Dataset (Tsanas et al., 2009) — **5,875 voice recordings** collected longitudinally from **42 early-to-mid-stage Parkinson's patients** (`28` male, `14` female; ages `36–85`), with **0 missing values**.
        * **Key Acoustic Biomarkers:**
          * *Nonlinear Dynamical Complexity:* Pitch Period Entropy (`PPE`), Detrended Fluctuation Analysis (`DFA`), Recurrence Period Density Entropy (`RPDE`).
          * *Noise & Harmonic Clarity:* Harmonics-to-Noise Ratio (`HNR`), Noise-to-Harmonics Ratio (`NHR`).
          * *Vocal Fold Micro-Instability:* Frequency perturbation (`Jitter(%)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`) and amplitude perturbation (`Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3/5/11`, `Shimmer:DDA`).
        * **Cohort Access & Scope Limits:** All 42 participants in the benchmark have diagnosed Parkinson's disease (no healthy control cohort); therefore, this model performs **severity staging among diagnosed patients** rather than population-wide screening.
        """
    )

    st.subheader("3. Leakage-Free Preprocessing & Feature Engineering")
    st.markdown(
        """
        * **Domain Feature Engineering (`18` $\\rightarrow$ `31` Features):** Created 13 clinically motivated interaction and nonlinear ratio features capturing how vocal degradation compounds with age and noise:
          * *Demographic-Acoustic Interactions:* `age_dfa`, `age_ppe`, `age_hnr`, `age_rpde` (accounts for age-related vocal fold atrophy).
          * *Acoustic Ratios & Composites:* `ppe_to_hnr` (pitch irregularity relative to harmonic clarity), log-transformed jitter/shimmer aggregates, and perturbation energy ratios.
        * **Strict Fold-Level Isolation:** All scaling (`StandardScaler`) and feature transformations are encapsulated inside a `scikit-learn` `Pipeline` and fit **exclusively on training folds** during cross-validation, guaranteeing zero test-set leakage.
        """
    )

    st.subheader("4. Baselines, Model Selection & Ablation Results")
    results_df = pd.DataFrame(
        {
            "Model / Experiment": [
                "Majority / Stratified Baseline",
                "Untuned Random Forest (18 raw features)",
                "Ablation: Remove Age & Sex (29 features)",
                "Ablation: Regression-to-Bins (RF Regressor)",
                "Ablation: Session-Level Averaging",
                "Final Tuned Random Forest (31 features, balanced)",
            ],
            "Macro F1": [0.2410, 0.3125, 0.3318, 0.3014, 0.3199, 0.4059],
            "Severe-Stage F1": [0.0000, 0.0400, 0.2140, 0.0921, 0.2285, 0.3236],
            "Key Takeaway": [
                "Ignores minority classes completely.",
                "Over-predicts Moderate (40.8% raw accuracy) while missing 96% of Severe cases.",
                "Proves age/sex baseline calibration is critical for interpreting pitch & noise.",
                "Mean-reversion pulls extreme predictions toward the center, collapsing Severe recall.",
                "Averaging across sessions smooths out acute vocal micro-tremor signals.",
                "8.1x gain in Severe F1 via class_weight='balanced', depth regularization & interactions.",
            ],
        }
    )
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("5. Evaluation Design & Metric Justification")
    st.markdown(
        """
        * **Patient-Grouped Validation (`StratifiedGroupKFold` by `subject#`):** Because each patient contributed ~140 recordings, a standard random split leaks the same speaker's vocal timbre into both train and test sets. Grouping by `subject#` evaluates generalization strictly across **42 unseen patients**.
        * **Why Macro F1 & Severe F1 over Raw Accuracy:** Under class imbalance, predicting `Moderate` for nearly every sample yields ~`41%` accuracy while failing clinically (`0.0400` Severe F1). **Macro F1 (`0.4059`)** weights all three clinical tiers equally, while **Severe F1 (`0.3236`)** directly tracks sensitivity to high-risk patients.
        """
    )

    st.subheader("6. Representative Error Analysis")
    st.markdown(
        """
        * **Threshold Boundary Ambiguity (`total_UPDRS` near `24` and `34`):** Most misclassifications occur within $\\pm 2.5$ UPDRS points of the discretization cutoffs, where continuous symptom progression crosses a hard label boundary without an abrupt shift in vocal acoustics.
        * **Precision–Recall Trade-off on `Moderate` Cases:** Introducing `class_weight="balanced"` and `max_depth=6` regularization dramatically improves detection of `Severe` cases (`0.0400` $\\rightarrow$ `0.3236`), while shifting some borderline `Moderate` recordings into adjacent `Mild` or `Severe` predictions.
        * **Inter-Speaker Vocal Variability:** Natural baseline differences in vocal pitch and breathiness across unseen speakers occasionally cause older patients with naturally lower `HNR` to be over-triaged.
        """
    )

    st.subheader("7. Limitations & Causal Caveats")
    st.warning(
        """
        * **Correlation, Not Causation:** Acoustic biomarkers reflect motor-speech manifestations associated with Parkinson's progression; model predictions do **not** imply that vocal changes cause UPDRS progression and cannot substitute for a neurologist's clinical assessment.
        * **Modality Scope:** Sustained-vowel phonation captures laryngeal/respiratory stability but does not observe non-speech motor symptoms (bradykinesia, limb rigidity, postural instability) or medication timing (`ON`/`OFF` levodopa states).
        """
    )

    st.subheader("8. Conclusion & Reproduction Steps")
    st.markdown(
        """
        **Conclusion:** Patient-grouped evaluation demonstrates that combining domain-engineered vocal interaction features (`age_dfa`, `age_ppe`, `ppe_to_hnr`) with class-balanced, depth-regularized Random Forests improves Severe-stage Parkinson's detection **8.1-fold** over an untuned baseline while maintaining honest cross-patient generalization.
        """
    )
    st.code(
        """# 1. Clone the repository
git clone https://github.com/bhuvi-iitgn/parkinsons-severity-classification.git
cd parkinsons-severity-classification

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt

# 3. Run standalone CLI inference or launch the Streamlit dashboard
python predict.py
streamlit run app.py""",
        language="bash",
    )