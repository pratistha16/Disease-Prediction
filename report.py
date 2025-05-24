import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🩺 Multi-Disease Diagnosis Assistant")
st.subheader("Step 1: Upload the patient's medical report (CSV format)")

uploaded_file = st.file_uploader("Upload report", type="csv")

# Models you have and their readable names
model_paths = {
    "bone_marrow": r"E:\sem 3\MSCP\models\bonemarror.pkl",
    "depression": r"E:\sem 3\MSCP\models\Depression.pkl",
    "wilson_disease": r"E:\sem 3\MSCP\models\rf_model.pkl"  # ✅ Update with correct model
}

# Matching disease names
disease_models = {
    "bone_marrow": "Bone Marrow Disease",
    "depression": "Depression",
    "wilson_disease": "Wilson Disease"
}

# Optional feature list paths (optional but smart)
feature_paths = {
    "wilson_disease": r"E:\sem 3\MSCP\models\feature_names.pkl"
}

# Session state storage
if "report_df" not in st.session_state:
    st.session_state.report_df = pd.DataFrame()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Patient report uploaded successfully!")
    st.dataframe(df)

    results = []

    for key, disease_name in disease_models.items():
        try:
            model = joblib.load(model_paths[key])

            # Optional: load expected feature list
            if key in feature_paths and os.path.exists(feature_paths[key]):
                expected_features = joblib.load(feature_paths[key])
                for col in expected_features:
                    if col not in df.columns:
                        df[col] = np.nan
                df_model = df[expected_features].copy()
            else:
                df_model = df.copy()

            # Impute missing numeric values
            df_model = df_model.fillna(df_model.mean(numeric_only=True))

            # Prediction
            prediction = model.predict(df_model)[0]
            prob = model.predict_proba(df_model)[0][1] if hasattr(model, "predict_proba") else (1.0 if prediction == 1 else 0.0)

            pred_text = f"Positive for {disease_name}" if str(prediction).lower() in ["positive", "yes", "1", "true"] else f"Negative for {disease_name}"

            results.append({
                "Disease": disease_name,
                "Prediction": pred_text,
                "Confidence (%)": round(prob * 100, 2)
            })

        except Exception as e:
            results.append({
                "Disease": disease_name,
                "Prediction": f"⚠️ Error: {str(e)}",
                "Confidence (%)": "N/A"
            })

    # Display and download
    report_df = pd.DataFrame(results)
    st.session_state.report_df = report_df

    st.subheader("🧾 Diagnostic Report")
    st.dataframe(report_df)

    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report as CSV", csv, "diagnostic_report.csv", "text/csv")

    if not any("Positive" in row for row in report_df["Prediction"]):
        st.success("🟢 No disease detected. The patient appears healthy.")
    else:
        st.warning("⚠️ Potential health concerns identified:")
        for row in report_df[report_df["Prediction"].str.contains("Positive")]["Disease"]:
            st.markdown(f"- **{row}**")
