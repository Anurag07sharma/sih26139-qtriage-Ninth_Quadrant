"""
train.py — Heart Disease: Full Hybrid Pipeline (Template for all diseases)
===========================================================================
This is the REFERENCE implementation. Other disease modules copy this pattern.

Pipeline:
  1. Load & split data
  2. Preprocess (impute, scale)
  3. PCA reduction (for quantum)
  4. Train classical baseline (XGBoost)
  5. Calibrate classical model
  6. Subsample for quantum training
  7. Train Quantum Committee (3 circuits)
  8. Build Confidence Cascade
  9. Evaluate: classical vs hybrid, per-tier breakdown
  10. SHAP explainability
  11. Save everything
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Add project root to path so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quantum_committee import QuantumCommittee
from cascade import ConfidenceCascade


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — CHANGE THESE PER DISEASE
# ═══════════════════════════════════════════════════════════════════════════════

DISEASE_NAME = "heart"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "heart.csv")
TARGET_COLUMN = "HeartDisease"  # Binary: 0 = no, 1 = yes
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_PCA_COMPONENTS = 6           # features fed to quantum circuits
QUANTUM_SUBSAMPLE = 250        # rows for quantum training (keep ≤300)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: LOAD & SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str):
    """Load CSV, separate features and target, encode categoricals."""
    print(f"\n[Stage 1] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}\n")

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    return X, y, label_encoders, list(X.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(X_train, X_test):
    """Impute missing values and scale features."""
    print("[Stage 2] Preprocessing (impute + scale)")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape:  {X_test_scaled.shape}\n")

    return X_train_scaled, X_test_scaled, imputer, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: PCA REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reduce_features(X_train, X_test, n_components):
    """PCA reduce for quantum circuits."""
    print(f"[Stage 3] PCA reduction → {n_components} components")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    variance = sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance retained: {variance:.1f}%")

    if variance < 85:
        print(f"  ⚠️  WARNING: Only {variance:.1f}% variance retained. "
              f"Consider increasing n_components.\n")
    else:
        print(f"  ✓ Good — above 85% threshold.\n")

    return X_train_pca, X_test_pca, pca


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: CLASSICAL BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def train_classical(X_train, y_train, X_test, y_test):
    """Train XGBoost, cross-validate, evaluate."""
    print("[Stage 4] Training classical baseline (XGBoost)")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test AUC-ROC:  {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return model, acc, auc


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: CALIBRATE CLASSICAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_classical(model, X_train, y_train):
    """Wrap model in CalibratedClassifierCV for reliable probabilities."""
    print("[Stage 5] Calibrating classical model (isotonic regression)")

    calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)

    print("  ✓ Model calibrated.\n")
    return calibrated


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: QUANTUM SUBSAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def quantum_subsample(X_train_pca, y_train, n_samples):
    """Stratified subsample for quantum training to keep runtime manageable."""
    print(f"[Stage 6] Subsampling {n_samples} rows for quantum training")

    if len(X_train_pca) <= n_samples:
        print(f"  Dataset already ≤ {n_samples} rows. Using all.\n")
        return X_train_pca, y_train

    _, X_sub, _, y_sub = train_test_split(
        X_train_pca, y_train,
        test_size=n_samples,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )

    print(f"  Subsample shape: {X_sub.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_sub, return_counts=True)))}\n")

    return X_sub, y_sub


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7: TRAIN QUANTUM COMMITTEE
# ═══════════════════════════════════════════════════════════════════════════════

def train_quantum(X_train_pca, y_train, n_features):
    """Train the 3-circuit Quantum Committee Machine."""
    print("[Stage 7] Training Quantum Committee Machine")

    committee = QuantumCommittee(n_features=n_features)
    committee.fit(X_train_pca, np.array(y_train))

    return committee


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8: BUILD CASCADE
# ═══════════════════════════════════════════════════════════════════════════════

def build_cascade(calibrated_model, committee, pca):
    """Wire together the Confidence Cascade."""
    print("[Stage 8] Building Confidence Cascade")

    cascade = ConfidenceCascade(
        classical_model=calibrated_model,
        committee=committee,
        pca_transform=pca.transform,
        high_threshold=0.90,
    )

    print("  ✓ Cascade ready.\n")
    return cascade


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9: EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(cascade, X_test, X_test_pca, y_test, classical_acc, classical_auc):
    """Run the full cascade on the test set and report results."""
    print("[Stage 9] Evaluating hybrid pipeline on test set")

    results = cascade.predict_batch(X_test, X_test_pca)

    # Extract predictions
    hybrid_preds = np.array([r["prediction"] for r in results])
    hybrid_acc = accuracy_score(y_test, hybrid_preds)

    # Tier breakdown
    summary = cascade.summary(results)

    print(f"\n  {'='*50}")
    print(f"  RESULTS COMPARISON")
    print(f"  {'='*50}")
    print(f"  Classical Accuracy:       {classical_acc:.4f}")
    print(f"  Hybrid (Cascade) Accuracy: {hybrid_acc:.4f}")
    print(f"  Delta:                     {hybrid_acc - classical_acc:+.4f}")
    print()
    print(f"  Tier Distribution:")
    for tier_name, count in summary["tier_distribution"].items():
        pct = count / summary["total_patients"] * 100
        print(f"    {tier_name}: {count} ({pct:.1f}%)")
    print()
    print(f"  Classical-only (Tier 1): {summary['classical_only_pct']}")
    print(f"  Quantum escalated:       {summary['quantum_escalated_pct']}")
    print(f"  Clinical alerts raised:  {summary['alerts_raised']} ({summary['alerts_pct']})")
    print(f"  {'='*50}\n")

    return results, hybrid_acc


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 10: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def explain_with_shap(model, X_test, feature_names, save_dir):
    """Generate SHAP explanations for the classical model."""
    print("[Stage 10] Generating SHAP explanations")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False
    )
    path = os.path.join(save_dir, f"{DISEASE_NAME}_shap_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary → {path}\n")

    return explainer, shap_values


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 11: SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

def save_all(model, calibrated_model, committee, pca, imputer, scaler,
             feature_names, explainer, save_dir):
    """Save all artifacts for the Streamlit dashboard to load."""
    print("[Stage 11] Saving models and artifacts")
    os.makedirs(save_dir, exist_ok=True)

    # Classical pipeline
    joblib.dump({
        "model": model,
        "calibrated_model": calibrated_model,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names,
        "explainer": explainer,
    }, os.path.join(save_dir, f"{DISEASE_NAME}_classical.joblib"))

    # Quantum committee
    committee.save(os.path.join(save_dir, f"{DISEASE_NAME}_committee.joblib"))

    print(f"  ✓ All artifacts saved to {save_dir}/\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  {DISEASE_NAME.upper()} DISEASE — FULL HYBRID PIPELINE")
    print("=" * 60)

    # 1. Load
    X, y, label_encoders, feature_names = load_data(DATA_PATH)

    # Split (BEFORE any preprocessing — no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # 2. Preprocess
    X_train_scaled, X_test_scaled, imputer, scaler = preprocess(X_train, X_test)

    # 3. PCA for quantum
    X_train_pca, X_test_pca, pca = reduce_features(
        X_train_scaled, X_test_scaled, N_PCA_COMPONENTS
    )

    # 4. Classical baseline
    model, classical_acc, classical_auc = train_classical(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # 5. Calibrate
    calibrated_model = calibrate_classical(model, X_train_scaled, y_train)

    # 6. Quantum subsample
    X_q_train, y_q_train = quantum_subsample(
        X_train_pca, y_train, QUANTUM_SUBSAMPLE
    )

    # 7. Train Quantum Committee
    committee = train_quantum(X_q_train, y_q_train, N_PCA_COMPONENTS)

    # 8. Build Cascade
    cascade = build_cascade(calibrated_model, committee, pca)

    # 9. Evaluate
    results, hybrid_acc = evaluate(
        cascade, X_test_scaled, X_test_pca, y_test,
        classical_acc, classical_auc
    )

    # 10. SHAP
    os.makedirs(MODEL_DIR, exist_ok=True)
    explainer, shap_values = explain_with_shap(
        model, X_test_scaled, feature_names, MODEL_DIR
    )

    # 11. Save
    save_all(
        model, calibrated_model, committee, pca, imputer, scaler,
        feature_names, explainer, MODEL_DIR
    )

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
