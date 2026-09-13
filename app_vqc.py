import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import pennylane as qml
import os

st.set_page_config(page_title="VQC Hybrid Disease Detection", layout="wide")

# ---------------------------------------------------------------------------
# QML Model Definition (Needs to match train_qml.py)
# ---------------------------------------------------------------------------
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def vqc_circuit(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    for layer in weights:
        for i in range(n_qubits):
            qml.RX(layer[i, 0], wires=i)
            qml.RY(layer[i, 1], wires=i)
            qml.RZ(layer[i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return qml.expval(qml.PauliZ(0))

class HybridQML(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (3, n_qubits, 3)}
        self.qml = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        
    def forward(self, x):
        x = self.qml(x)
        return (x + 1.0) / 2.0

# ---------------------------------------------------------------------------
# App UI
# ---------------------------------------------------------------------------
st.title("SIH26139: Hybrid Quantum Machine Learning Triage")
st.markdown("Comparing Classical Baselines vs. PennyLane VQC")

tab1, tab2 = st.tabs(["❤️ Heart Disease Triage", "🩸 Diabetes Triage"])

# --- Heart Disease Tab ---
with tab1:
    st.header("Patient Input")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50, key="h_age")
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0], key="h_sex")
        cpt = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], key="h_cpt")
    with col2:
        rbp = st.number_input("Resting BP", 80, 200, 120, key="h_rbp")
        mhr = st.number_input("Max Heart Rate", 60, 220, 150, key="h_mhr")
        chol = st.number_input("Cholesterol", 100, 400, 200, key="h_chol")

    if st.button("Evaluate Heart Risk", key="btn_heart"):
        # Load classical model
        if os.path.exists("models/heart_rf.pkl"):
            rf = joblib.load("models/heart_rf.pkl")
            input_df = pd.DataFrame([[age, sex, cpt, rbp, mhr, chol]], 
                                    columns=['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'max_heart_rate', 'cholesterol'])
            c_score = rf.predict_proba(input_df)[0][1]
        else:
            c_score = None
            
        # Load QML model
        if os.path.exists("models/heart_qml.pt"):
            qml_model = HybridQML()
            qml_model.load_state_dict(torch.load("models/heart_qml.pt"))
            qml_model.eval()
            
            # Pad to 8 dimensions for AmplitudeEmbedding
            padded_input = np.array([age, sex, cpt, rbp, mhr, chol, 0.0, 0.0], dtype=np.float32)
            padded_input = padded_input / np.linalg.norm(padded_input) # Normalize for amplitude
            t_input = torch.tensor(padded_input).unsqueeze(0)
            
            with torch.no_grad():
                q_score = qml_model(t_input).item()
        else:
            q_score = None

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            st.metric("Classical Risk Score (RF)", f"{c_score:.2%}" if c_score else "Model not found")
        with colB:
            st.metric("Hybrid Quantum Risk Score (VQC)", f"{q_score:.2%}" if q_score else "Model not found")

# --- Diabetes Tab ---
with tab2:
    st.header("Patient Input")
    col3, col4 = st.columns(2)
    with col3:
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key="d_bmi")
        genhlth = st.selectbox("GenHlth (1-5)", [1, 2, 3, 4, 5], key="d_gen")
        age_d = st.number_input("Age Bracket (1-13)", 1, 13, 5, key="d_age")
    with col4:
        highbp = st.selectbox("High BP (0=No, 1=Yes)", [0, 1], key="d_bp")
        highchol = st.selectbox("High Chol (0=No, 1=Yes)", [0, 1], key="d_chol")
        physhlth = st.number_input("PhysHlth (days)", 0, 30, 0, key="d_phys")
        
    # Appending income and education as they are in the context.md track specs
    income = st.selectbox("Income Bracket (1-8)", [1,2,3,4,5,6,7,8], key="d_inc")
    edu = st.selectbox("Education Level (1-6)", [1,2,3,4,5,6], key="d_edu")

    if st.button("Evaluate Diabetes Risk", key="btn_diab"):
        if os.path.exists("models/diabetes_xgb.pkl"):
            xgb = joblib.load("models/diabetes_xgb.pkl")
            input_df = pd.DataFrame([[bmi, genhlth, age_d, highbp, highchol, income, physhlth, edu]], 
                                    columns=['BMI', 'GenHlth', 'Age', 'HighBP', 'HighChol', 'Income', 'PhysHlth', 'Education'])
            # Reorder if necessary, assuming exact match
            c_score = xgb.predict_proba(input_df)[0][1]
            
            st.divider()
            st.metric("Classical Risk Score (XGBoost)", f"{c_score:.2%}")
            st.info("Quantum VQC for Diabetes not explicitly requested in Step 3, but classical is ready.")
        else:
            st.error("Classical model not found. Run train_classical.py")
