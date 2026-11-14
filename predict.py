import joblib
import numpy as np
import pandas as pd

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicates the exact Phase 4A domain feature engineering pipeline."""
    df_feat = df_raw.copy()
    df_feat['nonlinear_dysphonia'] = df_feat['RPDE'] * df_feat['DFA'] * df_feat['PPE']
    df_feat['jitter_shimmer_prod'] = df_feat['Jitter(%)'] * df_feat['Shimmer']
    df_feat['ppe_to_hnr'] = df_feat['PPE'] / (df_feat['HNR'] + 1e-5)
    df_feat['nhr_to_hnr'] = df_feat['NHR'] / (df_feat['HNR'] + 1e-5)
    df_feat['shimmer_apq_ratio'] = df_feat['Shimmer:APQ11'] / (df_feat['Shimmer:APQ3'] + 1e-6)
    df_feat['jitter_ppq_rap_ratio'] = df_feat['Jitter:PPQ5'] / (df_feat['Jitter:RAP'] + 1e-6)
    df_feat['age_ppe'] = df_feat['age'] * df_feat['PPE']
    df_feat['age_dfa'] = df_feat['age'] * df_feat['DFA']

    for col in ['Jitter(%)', 'Jitter(Abs)', 'Shimmer', 'NHR']:
        df_feat[f'log_{col}'] = np.log1p(df_feat[col])

    return df_feat

def predict_severity(sample_df: pd.DataFrame, model_path: str = "models/parkinsons_rf.joblib") -> pd.DataFrame:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    class_names = artifact["class_names"]

    X_engineered = engineer_features(sample_df)[feature_cols]
    preds = model.predict(X_engineered)
    probs = model.predict_proba(X_engineered)

    output = pd.DataFrame({
        "Predicted_Class": preds,
        "Severity_Label": [class_names[p] for p in preds],
        "Prob_Mild": np.round(probs[:, 0], 3),
        "Prob_Moderate": np.round(probs[:, 1], 3),
        "Prob_Severe": np.round(probs[:, 2], 3),
    })
    return output

if __name__ == "__main__":
    # Load 5 sample rows from the dataset to verify end-to-end inference
    raw_data = pd.read_csv("data/parkinsons_updrs.data")
    sample = raw_data.sample(5, random_state=42)
    
    predictions = predict_severity(sample)
    predictions["Actual_total_UPDRS"] = sample["total_UPDRS"].values
    print("=== Sample Patient Severity Predictions ===")
    print(predictions.to_string(index=False))