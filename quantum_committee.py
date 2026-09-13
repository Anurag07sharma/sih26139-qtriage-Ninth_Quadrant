"""
quantum_committee.py — The Quantum Committee Machine
=====================================================
Shared module for SIH26139. All disease modules import this.

Contains:
  - QuantumKernel: computes quantum kernel matrices for a given encoding
  - QuantumCommittee: 3 QSVMs (Angle, Amplitude, IQP) that vote together
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from tqdm import tqdm
import joblib
import os


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM KERNEL
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumKernel:
    """
    Computes a quantum kernel matrix using a specified feature encoding.

    The kernel value k(x1, x2) is the probability of measuring the |00...0⟩
    state after applying U†(x2) · U(x1), where U is the encoding circuit.
    This equals |⟨φ(x2)|φ(x1)⟩|² — the squared overlap of two quantum states.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    encoding : str
        One of "angle", "amplitude", or "iqp".
    """

    def __init__(self, n_qubits: int, encoding: str = "angle"):
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        dev = self.dev
        n = self.n_qubits
        enc = self.encoding

        if enc == "angle":
            @qml.qnode(dev)
            def circuit(x1, x2):
                qml.AngleEmbedding(x1, wires=range(n))
                qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n))
                return qml.probs(wires=range(n))
            return circuit

        elif enc == "amplitude":
            @qml.qnode(dev)
            def circuit(x1, x2):
                qml.AmplitudeEmbedding(x1, wires=range(n), normalize=True)
                qml.adjoint(qml.AmplitudeEmbedding)(
                    x2, wires=range(n), normalize=True
                )
                return qml.probs(wires=range(n))
            return circuit

        elif enc == "iqp":
            @qml.qnode(dev)
            def circuit(x1, x2):
                qml.IQPEmbedding(x1, wires=range(n), n_repeats=2)
                qml.adjoint(qml.IQPEmbedding)(
                    x2, wires=range(n), n_repeats=2
                )
                return qml.probs(wires=range(n))
            return circuit

        else:
            raise ValueError(f"Unknown encoding: {enc}. Use 'angle', 'amplitude', or 'iqp'.")

    def kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel between two samples."""
        return float(self._circuit(x1, x2)[0])

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray, desc: str = "") -> np.ndarray:
        """
        Compute the full kernel matrix K[i,j] = k(X1[i], X2[j]).

        Parameters
        ----------
        X1 : ndarray of shape (n1, n_features)
        X2 : ndarray of shape (n2, n_features)
        desc : str, optional
            Label for the progress bar.

        Returns
        -------
        K : ndarray of shape (n1, n2)
        """
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        total = n1 * n2
        label = f"  Kernel ({self.encoding})" if not desc else desc

        with tqdm(total=total, desc=label, leave=False) as pbar:
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel_value(X1[i], X2[j])
                    pbar.update(1)
        return K


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM COMMITTEE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumCommittee:
    """
    The Quantum Committee Machine — 3 QSVMs with fundamentally different
    quantum feature encodings that vote together on each patient.

    Circuit A (Angle):     captures individual feature magnitudes
    Circuit B (Amplitude): captures feature ratios & proportions
    Circuit C (IQP):       captures nonlinear feature interactions

    Parameters
    ----------
    n_features : int
        Number of PCA-reduced features. Used directly by Angle and IQP
        circuits (1 qubit per feature). Amplitude encoding pads to the
        next power of 2.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features

        # Angle & IQP: 1 qubit per feature
        self.n_qubits_angle = n_features
        self.n_qubits_iqp = n_features

        # Amplitude: needs 2^n dimensions, find smallest n
        self.n_qubits_amp = int(np.ceil(np.log2(max(n_features, 2))))
        self.amp_dim = 2 ** self.n_qubits_amp  # features will be padded to this

        # Build kernels
        self.kernel_angle = QuantumKernel(self.n_qubits_angle, "angle")
        self.kernel_amp = QuantumKernel(self.n_qubits_amp, "amplitude")
        self.kernel_iqp = QuantumKernel(self.n_qubits_iqp, "iqp")

        # SVMs (trained on precomputed kernel matrices for speed)
        self.svm_a = SVC(kernel="precomputed", probability=True, random_state=42)
        self.svm_b = SVC(kernel="precomputed", probability=True, random_state=42)
        self.svm_c = SVC(kernel="precomputed", probability=True, random_state=42)

        # Training data references (needed for prediction kernel computation)
        self._X_train = None
        self._X_train_padded = None

        # Cached training kernel matrices
        self._K_train_a = None
        self._K_train_b = None
        self._K_train_c = None

    def _pad_for_amplitude(self, X: np.ndarray) -> np.ndarray:
        """Pad feature vectors with zeros to reach 2^n_qubits_amp dimensions."""
        if X.shape[1] >= self.amp_dim:
            return X[:, :self.amp_dim]
        padding = np.zeros((X.shape[0], self.amp_dim - X.shape[1]))
        return np.hstack([X, padding])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train all three QSVMs on the same training data.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            PCA-reduced, scaled training features.
        y_train : ndarray of shape (n_samples,)
            Binary labels.
        """
        self._X_train = X_train.copy()
        X_padded = self._pad_for_amplitude(X_train)
        self._X_train_padded = X_padded

        n = len(X_train)
        print(f"\n{'='*60}")
        print(f"QUANTUM COMMITTEE TRAINING ({n} samples, {self.n_features} features)")
        print(f"{'='*60}")
        print(f"  Angle/IQP qubits: {self.n_qubits_angle}")
        print(f"  Amplitude qubits: {self.n_qubits_amp} (padded to {self.amp_dim} dims)")
        print(f"  Kernel matrix size: {n}×{n} = {n*n:,} evaluations per circuit")
        print()

        # Circuit A: Angle
        print("[1/3] Circuit A — Angle Encoding")
        self._K_train_a = self.kernel_angle.kernel_matrix(X_train, X_train)
        self.svm_a.fit(self._K_train_a, y_train)
        print("      ✓ Done\n")

        # Circuit B: Amplitude
        print("[2/3] Circuit B — Amplitude Encoding")
        self._K_train_b = self.kernel_amp.kernel_matrix(X_padded, X_padded)
        self.svm_b.fit(self._K_train_b, y_train)
        print("      ✓ Done\n")

        # Circuit C: IQP
        print("[3/3] Circuit C — IQP Encoding")
        self._K_train_c = self.kernel_iqp.kernel_matrix(X_train, X_train)
        self.svm_c.fit(self._K_train_c, y_train)
        print("      ✓ Done\n")

        print(f"{'='*60}")
        print("COMMITTEE READY")
        print(f"{'='*60}\n")

    def predict(self, X_test: np.ndarray) -> dict:
        """
        Get individual predictions from each committee member.

        Returns dict with keys 'angle', 'amplitude', 'iqp', each containing
        predicted labels for all test samples.
        """
        X_padded = self._pad_for_amplitude(X_test)

        K_a = self.kernel_angle.kernel_matrix(X_test, self._X_train, "Predict (Angle)")
        K_b = self.kernel_amp.kernel_matrix(X_padded, self._X_train_padded, "Predict (Amplitude)")
        K_c = self.kernel_iqp.kernel_matrix(X_test, self._X_train, "Predict (IQP)")

        return {
            "angle": self.svm_a.predict(K_a),
            "amplitude": self.svm_b.predict(K_b),
            "iqp": self.svm_c.predict(K_c),
        }

    def predict_proba(self, X_test: np.ndarray) -> dict:
        """Get probability estimates from each committee member."""
        X_padded = self._pad_for_amplitude(X_test)

        K_a = self.kernel_angle.kernel_matrix(X_test, self._X_train, "Proba (Angle)")
        K_b = self.kernel_amp.kernel_matrix(X_padded, self._X_train_padded, "Proba (Amplitude)")
        K_c = self.kernel_iqp.kernel_matrix(X_test, self._X_train, "Proba (IQP)")

        return {
            "angle": self.svm_a.predict_proba(K_a),
            "amplitude": self.svm_b.predict_proba(K_b),
            "iqp": self.svm_c.predict_proba(K_c),
        }

    def committee_vote(self, X_test: np.ndarray) -> list[dict]:
        """
        Run all three circuits and produce per-patient vote results.

        Returns
        -------
        list of dicts, one per test sample:
            {
                "votes": {"angle": 0/1, "amplitude": 0/1, "iqp": 0/1},
                "majority": 0 or 1,
                "agreement": "3/3" or "2/3",
                "unanimous": bool,
            }
        """
        preds = self.predict(X_test)

        results = []
        for i in range(len(X_test)):
            votes = {
                "angle": int(preds["angle"][i]),
                "amplitude": int(preds["amplitude"][i]),
                "iqp": int(preds["iqp"][i]),
            }
            vote_values = list(votes.values())
            majority = 1 if sum(vote_values) >= 2 else 0
            agreement = vote_values.count(majority)

            results.append({
                "votes": votes,
                "majority": majority,
                "agreement": f"{agreement}/3",
                "unanimous": agreement == 3,
            })

        return results

    def save(self, path: str):
        """Save the trained committee to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "n_features": self.n_features,
            "svm_a": self.svm_a,
            "svm_b": self.svm_b,
            "svm_c": self.svm_c,
            "X_train": self._X_train,
            "X_train_padded": self._X_train_padded,
            "K_train_a": self._K_train_a,
            "K_train_b": self._K_train_b,
            "K_train_c": self._K_train_c,
            "n_qubits_angle": self.n_qubits_angle,
            "n_qubits_amp": self.n_qubits_amp,
            "amp_dim": self.amp_dim,
        }, path)
        print(f"Committee saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QuantumCommittee":
        """Load a trained committee from disk."""
        data = joblib.load(path)
        committee = cls(data["n_features"])
        committee.svm_a = data["svm_a"]
        committee.svm_b = data["svm_b"]
        committee.svm_c = data["svm_c"]
        committee._X_train = data["X_train"]
        committee._X_train_padded = data["X_train_padded"]
        committee._K_train_a = data["K_train_a"]
        committee._K_train_b = data["K_train_b"]
        committee._K_train_c = data["K_train_c"]
        return committee
