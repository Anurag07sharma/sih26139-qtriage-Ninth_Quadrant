# Project Context: SIH26139 Hybrid Quantum ML Platform

## 1. Executive Summary
- **Problem Statement:** SIH26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection (Sponsor: Egreen Quanta).
- **Core Scope:** Focus exclusively on **Heart Disease** and **Diabetes**.
- **Objective:** Train Classical Baselines (Random Forest, XGBoost, PyTorch MLP) and a Parameterized Quantum Circuit (PennyLane VQC) to compare performance on early risk classification.

## 2. Directory Structure & File Paths
All scripts must adhere to the following directory layout:

- `prepare_data.py`: Data pipeline script.
- `data/raw/`: Untouched raw CSV source downloads.
- `data/cleaned/`: Unscaled clinical units (`*_train_clinical_units.csv`) for Streamlit widget min/max ranges.
- `data/processed/`: Standard-scaled CSV files (`heart_train.csv`, `heart_test.csv`, `diabetes_train.csv`, `diabetes_test.csv`).
- `data/quantum/`: Encoded `.npy` matrices (`*_angle.npy`, `*_amplitude.npy`, `*_labels.npy`) ready for PennyLane `qml.AngleEmbedding` or `qml.AmplitudeEmbedding`.

## 3. Selected Feature Specifications
### Heart Disease Tracks
1. **`heart` (UCI Combined Diagnostic):** 6 features (`age`, `sex`, `chest_pain_type`, `resting_bp_s`, `max_heart_rate`, `cholesterol`) | Target: `target` (1=Disease, 0=Normal).
2. **`heart_fhs` (Framingham 10-Year Prognostic):** 6 features (`age`, `sysBP`, `cigsPerDay`, `BMI`, `totChol`, `diaBP`) | Target: `TenYearCHD`.

### Diabetes Tracks
1. **`diabetes` (CDC BRFSS 2015 Triage):** 8 features (`BMI`, `GenHlth`, `Age`, `HighBP`, `HighChol`, `Income`, `PhysHlth`, `Education`) | Target: `target` (1=Diabetes/Prediabetes, 0=Healthy).
2. **`diabetes_pima` (Pima Indians Clinical):** 8 features (`Glucose`, `BMI`, `Age`, `Insulin`, `DiabetesPedigreeFunction`, `Pregnancies`, `BloodPressure`, `SkinThickness`) | Target: `target`.

## 4. Leakage Control Guarantees
- Imputers (`SimpleImputer(strategy='median')`), Scalers (`StandardScaler`), MinMax scalers (`[0, π]`), and PCA transformers are **fit on the training split only**.
- Test sets remain strictly unseen until final evaluation.