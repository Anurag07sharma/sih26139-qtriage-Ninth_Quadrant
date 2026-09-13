"""
cascade.py — Confidence Cascade + Disagreement Detection + 5-Tier Alert
========================================================================
Implements the full unified pipeline:
  Stage 1: Classical gatekeeper (calibrated XGBoost)
  Stage 2: Quantum Committee Machine (if classical is uncertain)
  Stage 3: Disagreement detection → 5-tier clinical alert
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from quantum_committee import QuantumCommittee


# ═══════════════════════════════════════════════════════════════════════════════
# TIER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

TIERS = {
    1: {"label": "Routine",          "color": "green",  "icon": "🟢"},
    2: {"label": "Resolved",         "color": "green",  "icon": "🟢"},
    3: {"label": "Committee Split",  "color": "yellow", "icon": "🟡"},
    4: {"label": "Paradigm Alert",   "color": "red",    "icon": "🔴"},
    5: {"label": "Severe Ambiguity", "color": "red",    "icon": "🔴"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE CASCADE
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceCascade:
    """
    Full Cascade → Committee → Alert pipeline.

    Parameters
    ----------
    classical_model : sklearn estimator
        A trained classifier (e.g., XGBClassifier). Will be wrapped in
        CalibratedClassifierCV for reliable probability estimates.
    committee : QuantumCommittee
        A trained QuantumCommittee instance.
    pca_transform : callable
        Function that takes raw features and returns PCA-reduced features
        for the quantum committee. Signature: pca_transform(X) -> X_reduced.
    high_threshold : float
        Classical confidence above this → Tier 1 (no quantum needed).
    """

    def __init__(
        self,
        classical_model,
        committee: QuantumCommittee,
        pca_transform=None,
        high_threshold: float = 0.90,
    ):
        self.classical_model = classical_model
        self.committee = committee
        self.pca_transform = pca_transform
        self.high_threshold = high_threshold

    @staticmethod
    def calibrate_model(model, X_train, y_train, cv=5):
        """
        Wrap a trained model in CalibratedClassifierCV so that
        predict_proba() returns real probabilities, not just scores.
        """
        calibrated = CalibratedClassifierCV(model, cv=cv, method="isotonic")
        calibrated.fit(X_train, y_train)
        return calibrated

    def predict_single(
        self,
        patient_features: np.ndarray,
        patient_features_pca: np.ndarray = None,
    ) -> dict:
        """
        Run the full cascade pipeline on a single patient.

        Parameters
        ----------
        patient_features : ndarray of shape (n_features_classical,)
            Full (uncompressed) feature vector for the classical model.
        patient_features_pca : ndarray of shape (n_features_quantum,), optional
            PCA-reduced features for the quantum committee.
            If None and self.pca_transform is set, it will be computed.

        Returns
        -------
        dict with keys:
            tier, prediction, confidence, model_path, alert,
            alert_type, message, votes (if quantum ran), classical_prediction
        """
        X_classical = patient_features.reshape(1, -1)

        # ── STAGE 1: Classical Gatekeeper ──────────────────────────────
        proba = self.classical_model.predict_proba(X_classical)[0]
        classical_confidence = float(max(proba))
        classical_pred = int(proba[1] > 0.5)

        # Tier 1: Classical is confident
        if classical_confidence >= self.high_threshold:
            return {
                "tier": 1,
                "tier_info": TIERS[1],
                "prediction": classical_pred,
                "confidence": classical_confidence,
                "model_path": "classical_only",
                "alert": False,
                "alert_type": None,
                "message": "Routine case — classical model confident.",
                "votes": None,
                "classical_prediction": classical_pred,
            }

        # ── STAGE 2: Quantum Committee ─────────────────────────────────
        if patient_features_pca is None and self.pca_transform is not None:
            patient_features_pca = self.pca_transform(X_classical).flatten()

        if patient_features_pca is None:
            raise ValueError(
                "Quantum committee requires PCA-reduced features. "
                "Provide patient_features_pca or set pca_transform."
            )

        X_pca = patient_features_pca.reshape(1, -1)
        vote_results = self.committee.committee_vote(X_pca)[0]

        votes = vote_results["votes"]
        committee_pred = vote_results["majority"]
        unanimous = vote_results["unanimous"]
        agreement_str = vote_results["agreement"]

        # ── STAGE 3: Disagreement Detection ────────────────────────────
        agrees_with_classical = (committee_pred == classical_pred)

        if unanimous and agrees_with_classical:
            # Tier 2: Both paradigms agree
            tier = 2
            alert = False
            alert_type = None
            message = (
                "Hard case resolved — classical and all 3 quantum circuits agree."
            )

        elif not unanimous and agrees_with_classical:
            # Tier 3: Committee split, but majority agrees with classical
            tier = 3
            alert = True
            alert_type = "committee_split"
            dissenters = [k for k, v in votes.items() if v != committee_pred]
            message = (
                f"⚠️ Quantum committee split ({agreement_str}). "
                f"{', '.join(dissenters)} encoding disagrees. "
                f"Majority aligns with classical model. "
                f"Recommend additional diagnostic tests."
            )

        elif unanimous and not agrees_with_classical:
            # Tier 4: All quantum circuits disagree with classical
            tier = 4
            alert = True
            alert_type = "cross_paradigm"
            message = (
                "🚨 CROSS-PARADIGM ALERT: All three quantum circuits "
                "unanimously disagree with the classical model. "
                "This patient's profile occupies a different decision region "
                "in quantum feature space. Specialist review highly recommended."
            )

        else:
            # Tier 5: Committee split AND disagrees with classical
            tier = 5
            alert = True
            alert_type = "severe_ambiguity"
            message = (
                "🚨 SEVERE AMBIGUITY: Classical model is uncertain, "
                "quantum committee is internally split, AND quantum majority "
                "disagrees with classical. This patient's profile is genuinely "
                "ambiguous across all mathematical frameworks. "
                "Full diagnostic workup strongly recommended."
            )

        return {
            "tier": tier,
            "tier_info": TIERS[tier],
            "prediction": committee_pred,
            "confidence": classical_confidence,
            "model_path": f"cascade → committee ({agreement_str})",
            "alert": alert,
            "alert_type": alert_type,
            "message": message,
            "votes": votes,
            "classical_prediction": classical_pred,
        }

    def predict_batch(
        self,
        X_classical: np.ndarray,
        X_pca: np.ndarray = None,
    ) -> list[dict]:
        """
        Run the cascade on multiple patients.

        Parameters
        ----------
        X_classical : ndarray of shape (n_patients, n_features_classical)
        X_pca : ndarray of shape (n_patients, n_features_quantum), optional

        Returns
        -------
        list of result dicts (one per patient)
        """
        if X_pca is None and self.pca_transform is not None:
            X_pca = self.pca_transform(X_classical)

        results = []
        for i in range(len(X_classical)):
            pca_row = X_pca[i] if X_pca is not None else None
            result = self.predict_single(X_classical[i], pca_row)
            results.append(result)

        return results

    def summary(self, results: list[dict]) -> dict:
        """
        Summarize batch predictions by tier distribution.
        Useful for evaluation and reporting.
        """
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in results:
            tier_counts[r["tier"]] += 1

        total = len(results)
        quantum_used = sum(1 for r in results if r["tier"] >= 2)
        alerts = sum(1 for r in results if r["alert"])

        return {
            "total_patients": total,
            "tier_distribution": {
                f"Tier {k} ({TIERS[k]['icon']} {TIERS[k]['label']})": v
                for k, v in tier_counts.items()
            },
            "classical_only_pct": f"{tier_counts[1] / total * 100:.1f}%",
            "quantum_escalated_pct": f"{quantum_used / total * 100:.1f}%",
            "alerts_raised": alerts,
            "alerts_pct": f"{alerts / total * 100:.1f}%",
        }
