"""
train.py — Breast Cancer: Full Hybrid Pipeline
================================================
Wisconsin Diagnostic dataset (569 rows, 30 features).
30 features → PCA to 6 for quantum.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quantum_committee import QuantumCommittee
from cascade import ConfidenceCascade


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — BREAST CANCER
# ═══════════════════════════════════════════════════════════════════════════════

DISEASE_NAME = "breast_cancer"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "breast_cancer.csv")
TARGET_COLUMN = "diagnosis"      # M = malignant (1), B = benign (0)
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_PCA_COMPONENTS = 6             # 30 features → 6 for quantum
QUANTUM_SUBSAMPLE = 250          # 455 train rows, subsample to 250
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def load_data(path: str):
    print(f"\n[Stage 1] Loading data from {path}")
    df = pd.read_csv(path)

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    if "Unnamed: 32" in df.columns:
        df = df.drop("Unnamed: 32", axis=1)

    # Encode diagnosis: M=1, B=0
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "M").astype(int)

    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}\n")

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y, {}, list(X.columns)


def preprocess(X_train, X_test):
    print("[Stage 2] Preprocessing (impute + scale)")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}\n")
    return X_train_scaled, X_test_scaled, imputer, scaler


def reduce_features(X_train, X_test, n_components):
    print(f"[Stage 3] PCA reduction → {n_components} components")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    variance = sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance retained: {variance:.1f}%\n")
    return X_train_pca, X_test_pca, pca


def train_classical(X_train, y_train, X_test, y_test):
    print("[Stage 4] Training classical baseline (XGBoost)")
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_SEED, use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test AUC-ROC:  {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    return model, acc, auc


def main():
    print("=" * 60)
    print(f"  {DISEASE_NAME.upper()} — FULL HYBRID PIPELINE")
    print("=" * 60)

    X, y, _, feature_names = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_train_scaled, X_test_scaled, imputer, scaler = preprocess(X_train, X_test)
    X_train_pca, X_test_pca, pca = reduce_features(X_train_scaled, X_test_scaled, N_PCA_COMPONENTS)
    model, classical_acc, classical_auc = train_classical(X_train_scaled, y_train, X_test_scaled, y_test)

    print("[Stage 5] Calibrating classical model")
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_train_scaled, y_train)
    print("  ✓ Calibrated.\n")

    print(f"[Stage 6] Subsampling {QUANTUM_SUBSAMPLE} rows for quantum")
    if len(X_train_pca) <= QUANTUM_SUBSAMPLE:
        X_q, y_q = X_train_pca, np.array(y_train)
    else:
        _, X_q, _, y_q = train_test_split(
            X_train_pca, y_train, test_size=QUANTUM_SUBSAMPLE,
            stratify=y_train, random_state=RANDOM_SEED,
        )
        y_q = np.array(y_q)
    print(f"  Subsample: {X_q.shape}\n")

    print("[Stage 7] Training Quantum Committee Machine")
    committee = QuantumCommittee(n_features=N_PCA_COMPONENTS)
    committee.fit(X_q, y_q)

    print("[Stage 8] Building Confidence Cascade")
    cascade = ConfidenceCascade(
        classical_model=calibrated_model, committee=committee,
        pca_transform=pca.transform, high_threshold=0.90,
    )
    print("  ✓ Cascade ready.\n")

    print("[Stage 9] Evaluating hybrid pipeline")
    results = cascade.predict_batch(X_test_scaled, X_test_pca)
    hybrid_preds = np.array([r["prediction"] for r in results])
    hybrid_acc = accuracy_score(y_test, hybrid_preds)
    summary = cascade.summary(results)
    print(f"  Classical Accuracy: {classical_acc:.4f}")
    print(f"  Hybrid Accuracy:    {hybrid_acc:.4f}")
    print(f"  Delta:              {hybrid_acc - classical_acc:+.4f}\n")

    print("[Stage 10] Generating SHAP explanations")
    os.makedirs(MODEL_DIR, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{DISEASE_NAME}_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("[Stage 11] Saving models")
    joblib.dump({
        "model": model, "calibrated_model": calibrated_model,
        "imputer": imputer, "scaler": scaler, "pca": pca,
        "feature_names": feature_names, "explainer": explainer,
    }, os.path.join(MODEL_DIR, f"{DISEASE_NAME}_classical.joblib"))
    committee.save(os.path.join(MODEL_DIR, f"{DISEASE_NAME}_committee.joblib"))

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
