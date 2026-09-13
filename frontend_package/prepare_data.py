#!/usr/bin/env python3
"""
SIH26139 - Data Preparation & Preprocessing Pipeline (Heart & Diabetes Focus)
================================================================================
Output:
  - data/processed/  : Cleaned, train-fit scaled CSV datasets for ML training
  - data/cleaned/    : Raw unscaled clinical-unit CSVs for Streamlit UI forms
  - data/quantum/    : Pre-encoded .npy matrices for PennyLane / QML models
  - data/processed/metadata.json : SHA-256 audit trail & feature rankings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "cleaned"
PROC = ROOT / "data" / "processed"
QUANT = ROOT / "data" / "quantum"

for d in (RAW, CLEAN, PROC, QUANT):
    d.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "heart_uci_combined": dict(
        name="Heart Disease Dataset (Comprehensive / 5-cohort merge)",
        primary_source="UCI ML Repository (id=45) + Statlog(Heart)",
        download_url="https://raw.githubusercontent.com/rikhuijzer/heart-disease-dataset/main/heart-disease-dataset.csv",
        fname="heart_combined_1190.csv",
        expected_shape=(1190, 12),
    ),
    "framingham": dict(
        name="Framingham Heart Study - Teaching Subset (10-year CHD risk)",
        primary_source="Framingham Heart Study (NHLBI / BioLINCC)",
        download_url="https://raw.githubusercontent.com/plaguedoc000/framingham-heart-study-dataset/main/framingham.csv",
        fname="framingham.csv",
        expected_shape=(4240, 16),
    ),
    "brfss2015": dict(
        name="Diabetes Health Indicators - CDC BRFSS 2015",
        primary_source="CDC Behavioral Risk Factor Surveillance System 2015",
        download_url="https://raw.githubusercontent.com/laninh-tech/Diabetes-Health-Risk-Analysis/main/data/diabetes_012_health_indicators_BRFSS2015.csv",
        fname="brfss2015_diabetes_012.csv",
        expected_shape=(253680, 22),
    ),
    "pima": dict(
        name="Pima Indians Diabetes Database",
        primary_source="NIDDK / UCI ML Repository (id=34)",
        download_url="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        fname="pima_diabetes.csv",
        expected_shape=(768, 9),
    ),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_all(offline: bool = False) -> dict:
    import urllib.request
    report = {}
    for key, s in SOURCES.items():
        dest = RAW / s["fname"]
        if not dest.exists():
            if offline:
                raise FileNotFoundError(f"--offline flag set but {dest} is missing.")
            print(f"  Downloading {key} ...")
            urllib.request.urlretrieve(s["download_url"], dest)

        df = pd.read_csv(dest)
        ok = tuple(df.shape) == tuple(s["expected_shape"])
        report[key] = dict(
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            shape_matches_documented_original=bool(ok),
            sha256=sha256(dest),
            bytes=dest.stat().st_size,
        )
        status = "OK" if ok else "SHAPE MISMATCH"
        print(f"  [{status}] {key:22s} {df.shape} sha256={report[key]['sha256'][:16]}")
        if not ok:
            raise ValueError(f"{key}: got {df.shape}, documented {s['expected_shape']}")
    return report

def ensemble_feature_selection(X: pd.DataFrame, y: np.ndarray, k: int, max_rows: int = 20000, must_keep: list[str] | None = None) -> dict:
    rng = np.random.RandomState(SEED)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        Xs, ys = X.iloc[idx], y[idx]
    else:
        Xs, ys = X, y

    feats = list(X.columns)
    ranks = {}

    imp = SimpleImputer(strategy="median").fit(Xs)
    Xs_clean = pd.DataFrame(imp.transform(Xs), columns=feats, index=Xs.index)

    mi = mutual_info_classif(Xs_clean, ys, random_state=SEED)
    ranks["mutual_info"] = pd.Series(mi, index=feats)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample")
    rf.fit(Xs_clean, ys)
    ranks["rf_importance"] = pd.Series(rf.feature_importances_, index=feats)

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, eval_metric="logloss")
        xgb.fit(Xs_clean, ys)
        ranks["xgb_gain"] = pd.Series(xgb.feature_importances_, index=feats)

    tr, va = train_test_split(np.arange(len(Xs_clean)), test_size=0.25, stratify=ys, random_state=SEED)
    rf2 = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf2.fit(Xs_clean.iloc[tr], ys[tr])
    pi = permutation_importance(rf2, Xs_clean.iloc[va], ys[va], n_repeats=5, random_state=SEED, n_jobs=-1)
    ranks["permutation"] = pd.Series(pi.importances_mean, index=feats)

    if HAS_SHAP and HAS_XGB:
        sv = shap.TreeExplainer(xgb).shap_values(Xs_clean.iloc[: min(2000, len(Xs_clean))])
        if isinstance(sv, list):
            sv = sv[-1]
        ranks["shap"] = pd.Series(np.abs(sv).mean(axis=0), index=feats)

    fold_ranks = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs_clean, ys)
    for tr_i, _ in splitter:
        m = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
        m.fit(Xs_clean.iloc[tr_i], ys[tr_i])
        fold_ranks.append(pd.Series(m.feature_importances_, index=feats).rank(ascending=False))
    fold_ranks = pd.concat(fold_ranks, axis=1)
    ranks["cv_stability"] = -fold_ranks.mean(axis=1)

    rank_df = pd.DataFrame({k_: v.rank(ascending=False) for k_, v in ranks.items()})
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values("mean_rank")

    corr = Xs_clean.corr(method="spearman").abs()
    ordered = list(rank_df.index)
    kept, dropped = [], {}
    for f in ordered:
        red = [g for g in kept if corr.loc[f, g] > 0.90]
        if red:
            dropped[f] = dict(redundant_with=red[0], spearman=round(float(corr.loc[f, red[0]]), 4))
        else:
            kept.append(f)

    if must_keep:
        for f in reversed(must_keep):
            if f in X.columns:
                if f in kept:
                    kept.remove(f)
                kept.insert(0, f)

    return dict(selected=kept[:k], ranking=rank_df.round(4).to_dict(orient="index"), dropped_as_redundant=dropped)

def make_quantum(Xtr_sel: pd.DataFrame, n_samples: int, y_tr: np.ndarray, tag: str) -> dict:
    n = len(Xtr_sel)
    take = min(n_samples, n)
    idx = train_test_split(np.arange(n), train_size=take, stratify=y_tr, random_state=SEED)[0] if take < n else np.arange(n)
    Xq, yq = Xtr_sel.iloc[idx], y_tr[idx]

    mm_pi = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xtr_sel)
    angle = mm_pi.transform(Xq)

    mm01 = MinMaxScaler(feature_range=(0.0, 1.0)).fit(Xtr_sel)
    A = mm01.transform(Xq)
    d = A.shape[1]
    n_qubits = int(np.ceil(np.log2(d))) if d > 1 else 1
    pad_to = 2**n_qubits
    n_pad = pad_to - d

    if n_pad > 0:
        A = np.hstack([A, np.zeros((len(A), n_pad))])

    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    amp = A / norms

    np.save(QUANT / f"{tag}_angle.npy", angle.astype(np.float64))
    np.save(QUANT / f"{tag}_iqp.npy", angle.astype(np.float64))
    np.save(QUANT / f"{tag}_amplitude.npy", amp.astype(np.float64))
    np.save(QUANT / f"{tag}_labels.npy", yq.astype(np.int64))
    np.save(QUANT / f"{tag}_amplitude_norms.npy", norms.astype(np.float64))

    return dict(n_samples=int(take), dims=int(d), n_qubits=int(n_qubits), padded_dimensions=int(pad_to))

def finalize(tag: str, Xtr: pd.DataFrame, Xte: pd.DataFrame, ytr: np.ndarray, yte: np.ndarray, sel: list[str], meta: dict):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i = pd.DataFrame(imp.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=Xte.columns, index=Xte.index)

    sc = StandardScaler().fit(Xtr_i[sel])
    tr_out = pd.DataFrame(sc.transform(Xtr_i[sel]), columns=sel)
    te_out = pd.DataFrame(sc.transform(Xte_i[sel]), columns=sel)
    tr_out["target"] = ytr
    te_out["target"] = yte

    tr_out.to_csv(PROC / f"{tag}_train.csv", index=False)
    te_out.to_csv(PROC / f"{tag}_test.csv", index=False)

    Xtr_i[sel].assign(target=ytr).to_csv(CLEAN / f"{tag}_train_clinical_units.csv", index=False)
    Xte_i[sel].assign(target=yte).to_csv(CLEAN / f"{tag}_test_clinical_units.csv", index=False)

    meta["quantum"] = make_quantum(Xtr_i[sel], 300, ytr, tag)
    meta["train_rows"] = int(len(tr_out))
    meta["test_rows"] = int(len(te_out))
    meta["selected_features"] = sel
    print(f"    -> {tag}: Train={len(tr_out)}, Test={len(te_out)}, Features={sel}")
    return meta

def build_heart():
    s = SOURCES["heart_uci_combined"]
    df = pd.read_csv(RAW / s["fname"]).drop_duplicates().reset_index(drop=True)
    df.loc[df.cholesterol == 0, "cholesterol"] = np.nan
    df.loc[df.resting_bp_s == 0, "resting_bp_s"] = np.nan
    y = df.pop("target").astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.20, stratify=y, random_state=SEED)
    user_features = ["age", "sex", "chest_pain_type", "resting_bp_s", "max_heart_rate", "cholesterol"]
    fs = ensemble_feature_selection(Xtr, ytr, k=6, must_keep=user_features)
    return finalize("heart", Xtr, Xte, ytr, yte, fs["selected"], dict(dataset=s["name"], feature_selection=fs))

def build_heart_framingham():
    s = SOURCES["framingham"]
    df = pd.read_csv(RAW / s["fname"])
    y = df.pop("TenYearCHD").astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.20, stratify=y, random_state=SEED)
    user_features = ["age", "sysBP", "cigsPerDay", "BMI", "totChol", "diaBP"]
    fs = ensemble_feature_selection(Xtr, ytr, k=6, must_keep=user_features)
    return finalize("heart_fhs", Xtr, Xte, ytr, yte, fs["selected"], dict(dataset=s["name"], feature_selection=fs))

def build_diabetes_brfss(n_target=100_000):
    s = SOURCES["brfss2015"]
    df = pd.read_csv(RAW / s["fname"])
    y_full = (df.pop("Diabetes_012") > 0).astype(int).values
    idx = train_test_split(np.arange(len(df)), train_size=n_target, stratify=y_full, random_state=SEED)[0]
    X, y = df.iloc[idx].reset_index(drop=True), y_full[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    user_features = ["BMI", "GenHlth", "Age", "HighBP", "HighChol", "Income", "PhysHlth", "Education"]
    fs = ensemble_feature_selection(Xtr, ytr, k=8, must_keep=user_features)
    return finalize("diabetes", Xtr, Xte, ytr, yte, fs["selected"], dict(dataset=s["name"], feature_selection=fs))

def build_diabetes_pima():
    s = SOURCES["pima"]
    df = pd.read_csv(RAW / s["fname"])
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df.loc[df[c] == 0, c] = np.nan
    y = df.pop("Outcome").astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.20, stratify=y, random_state=SEED)
    user_features = ["Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction", "Pregnancies", "BloodPressure", "SkinThickness"]
    fs = ensemble_feature_selection(Xtr, ytr, k=8, must_keep=user_features)
    return finalize("diabetes_pima", Xtr, Xte, ytr, yte, fs["selected"], dict(dataset=s["name"], feature_selection=fs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print("\n[1/3] Downloading & Verifying Data...")
    dl_report = download_all(args.offline)

    print("\n[2/3] Processing Datasets & Quantum Arrays...")
    meta = {}
    for tag, fn in [("heart", build_heart), ("heart_fhs", build_heart_framingham), ("diabetes", build_diabetes_brfss), ("diabetes_pima", build_diabetes_pima)]:
        print(f"  Building {tag} ...")
        meta[tag] = fn()

    print("\n[3/3] Saving Metadata Audit Trail...")
    with open(PROC / "metadata.json", "w") as f:
        json.dump(dict(project="SIH26139", download_verification=dl_report, datasets=meta), f, indent=2, default=str)
    print("SUCCESS: Datasets ready for QML training.\n")

if __name__ == "__main__":
    main()