"""
app.py — Streamlit Dashboard
==============================
Unified multi-disease clinical decision support UI.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

from triage.router import triage, DISEASE_MODULES, get_all_features
from cascade import ConfidenceCascade, TIERS
from quantum_committee import QuantumCommittee

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quantum-Enhanced Disease Prediction",
    page_icon="🧬",
    layout="wide",
)

MODEL_DIR = "models"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Load a saved disease pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_disease_pipeline(disease_name: str):
    """Load classical + quantum pipeline for a disease from saved joblib files."""
    classical_path = os.path.join(MODEL_DIR, f"{disease_name}_classical.joblib")
    committee_path = os.path.join(MODEL_DIR, f"{disease_name}_committee.joblib")

    if not os.path.exists(classical_path):
        return None

    classical = joblib.load(classical_path)

    committee = None
    if os.path.exists(committee_path):
        committee = QuantumCommittee.load(committee_path)

    return {
        "classical": classical,
        "committee": committee,
    }


def run_prediction(disease_name: str, patient_data: dict) -> dict:
    """Run the full cascade pipeline for a disease."""
    pipeline = load_disease_pipeline(disease_name)
    if pipeline is None:
        return {"error": f"No trained model found for {disease_name}"}

    classical = pipeline["classical"]
    committee = pipeline["committee"]

    # Prepare features in the correct order
    feature_names = classical["feature_names"]
    features = np.array([[patient_data.get(f, 0) for f in feature_names]])

    # Preprocess
    features_imp = classical["imputer"].transform(features)
    features_scaled = classical["scaler"].transform(features_imp)

    # PCA for quantum
    pca = classical["pca"]
    features_pca = pca.transform(features_scaled)

    if committee is not None:
        # Full cascade pipeline
        cascade = ConfidenceCascade(
            classical_model=classical["calibrated_model"],
            committee=committee,
            pca_transform=pca.transform,
            high_threshold=0.90,
        )
        result = cascade.predict_single(features_scaled[0], features_pca[0])
    else:
        # Classical-only fallback
        proba = classical["calibrated_model"].predict_proba(features_scaled)[0]
        result = {
            "tier": 1,
            "tier_info": TIERS[1],
            "prediction": int(proba[1] > 0.5),
            "confidence": float(max(proba)),
            "model_path": "classical_only",
            "alert": False,
            "alert_type": None,
            "message": "Classical model prediction.",
            "votes": None,
            "classical_prediction": int(proba[1] > 0.5),
        }

    # SHAP explanation
    explainer = classical.get("explainer")
    if explainer is not None:
        shap_values = explainer.shap_values(features_scaled)
        result["shap_values"] = shap_values[0]
        result["feature_names"] = feature_names

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🧬 Quantum-Enhanced Clinical Decision Support")
    st.markdown(
        "Enter patient data below. The system will automatically determine "
        "which disease modules can evaluate the patient based on available features."
    )

    st.divider()

    # ── Patient Input Form ─────────────────────────────────────────────────
    st.header("📋 Patient Data Entry")

    col1, col2, col3 = st.columns(3)

    patient_data = {}

    with col1:
        st.subheader("General")
        patient_data["age"] = st.number_input("Age", min_value=0, max_value=120, value=0, key="age")
        patient_data["sex"] = st.selectbox("Sex", ["", "Male", "Female"], key="sex")
        if patient_data["sex"] == "Male":
            patient_data["sex"] = 1
        elif patient_data["sex"] == "Female":
            patient_data["sex"] = 0
        else:
            patient_data["sex"] = None

        st.subheader("Cardiovascular")
        patient_data["chest_pain_type"] = st.selectbox(
            "Chest Pain Type",
            ["", "Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
            key="cpt"
        )
        if patient_data["chest_pain_type"]:
            patient_data["chest_pain_type"] = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(patient_data["chest_pain_type"])
        else:
            patient_data["chest_pain_type"] = None

        patient_data["resting_bp"] = st.number_input("Resting BP (mmHg)", min_value=0, max_value=300, value=0, key="rbp")
        patient_data["cholesterol"] = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=0, key="chol")
        patient_data["max_hr"] = st.number_input("Max Heart Rate", min_value=0, max_value=250, value=0, key="mhr")

    with col2:
        st.subheader("Metabolic")
        patient_data["glucose"] = st.number_input("Blood Glucose (mg/dl)", min_value=0, max_value=500, value=0, key="gluc")
        patient_data["bmi"] = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1, key="bmi")
        patient_data["insulin"] = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=0, key="ins")
        patient_data["HbA1c_level"] = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=0.0, step=0.1, key="hba1c")

        st.subheader("Other")
        patient_data["fasting_bs"] = st.selectbox("Fasting BS > 120?", ["", "Yes", "No"], key="fbs")
        if patient_data["fasting_bs"] == "Yes":
            patient_data["fasting_bs"] = 1
        elif patient_data["fasting_bs"] == "No":
            patient_data["fasting_bs"] = 0
        else:
            patient_data["fasting_bs"] = None

    with col3:
        st.subheader("Cell Measurements")
        patient_data["radius_mean"] = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=0.0, step=0.01, key="rm")
        patient_data["texture_mean"] = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=0.0, step=0.01, key="tm")
        patient_data["perimeter_mean"] = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=0.0, step=0.01, key="pm")

        st.subheader("Voice (Neurological)")
        patient_data["mdvp_fo"] = st.number_input("MDVP:Fo (Hz)", min_value=0.0, max_value=300.0, value=0.0, step=0.01, key="fo")
        patient_data["mdvp_jitter"] = st.number_input("MDVP:Jitter (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.001, key="jit")
        patient_data["mdvp_shimmer"] = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.0, step=0.001, key="shim")

    # Clean zero-values as "not entered"
    for key in list(patient_data.keys()):
        if patient_data[key] == 0 or patient_data[key] == 0.0:
            patient_data[key] = None

    st.divider()

    # ── Analyze Button ─────────────────────────────────────────────────────
    if st.button("🔍 Analyze Patient", type="primary", use_container_width=True):

        # Run triage
        triage_result = triage(patient_data)
        eligible = triage_result["eligible"]
        ineligible = triage_result["ineligible"]

        st.header("📊 Analysis Results")

        if not eligible:
            st.warning("No disease modules could run — not enough features entered.")
        else:
            st.success(f"**Eligible modules:** {', '.join(DISEASE_MODULES[d]['display_name'] for d in eligible)}")

        # ── Results for each eligible module ───────────────────────────
        for disease in eligible:
            display = DISEASE_MODULES[disease]["display_name"]

            with st.expander(f"{display} — Results", expanded=True):
                result = run_prediction(disease, patient_data)

                if "error" in result:
                    st.error(result["error"])
                    continue

                tier = result["tier"]
                tier_info = result["tier_info"]
                prediction = result["prediction"]
                confidence = result["confidence"]

                # Risk bar
                risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_color = "🔴" if prediction == 1 else "🟢"

                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    st.metric("Risk Assessment", f"{risk_color} {risk_label}")
                    st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                with col_r2:
                    st.metric("Alert Tier", f"{tier_info['icon']} Tier {tier}")
                    st.caption(tier_info["label"])

                # Pipeline trace
                st.markdown(f"**Model Path:** `{result['model_path']}`")

                # Alert message
                if result["alert"]:
                    st.warning(result["message"])
                else:
                    st.info(result["message"])

                # Quantum votes (if committee ran)
                if result["votes"]:
                    st.markdown("**Quantum Committee Votes:**")
                    vote_cols = st.columns(3)
                    for i, (circuit, vote) in enumerate(result["votes"].items()):
                        label = "HIGH RISK" if vote == 1 else "LOW RISK"
                        icon = "🔴" if vote == 1 else "🟢"
                        vote_cols[i].metric(
                            f"Circuit {circuit.title()}",
                            f"{icon} {label}"
                        )

                # SHAP explanation
                if "shap_values" in result:
                    st.markdown("**SHAP Explanation (Top Factors):**")
                    shap_df = pd.DataFrame({
                        "Feature": result["feature_names"],
                        "Impact": result["shap_values"],
                    })
                    shap_df["Abs_Impact"] = shap_df["Impact"].abs()
                    shap_df = shap_df.sort_values("Abs_Impact", ascending=False).head(8)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#d726ff" if v > 0 else "#26d7ff" for v in shap_df["Impact"]]
                    ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
                    ax.set_xlabel("SHAP Impact (positive = increases risk)")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # ── Ineligible modules ─────────────────────────────────────────
        if ineligible:
            st.divider()
            st.subheader("Modules Not Evaluated")
            for disease, reason in ineligible.items():
                display = DISEASE_MODULES[disease]["display_name"]
                st.caption(f"⛔ {display}: {reason}")


if __name__ == "__main__":
    main()
