"""
models.py
---------------------------------------------------------------
Local, free, no-API machine learning layer.

Heterogeneous ensemble (Logistic Regression, Random Forest, Gradient Boosting)
with isotonic/sigmoid calibration, predictive entropy calculation, and epistemic
uncertainty estimation.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.features import RAW_FEATURES, PHENOTYPE_COLUMNS
from src.schemas import PredictionResult

FEATURE_COLUMNS = RAW_FEATURES + PHENOTYPE_COLUMNS
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def compute_binary_entropy(prob: float, eps: float = 1e-7) -> float:
    """Computes binary Shannon entropy in bits for probability p in [0, 1].
    H(p) = -p*log2(p) - (1-p)*log2(1-p).
    0.0 = maximum certainty (p=0 or p=1), 1.0 = maximum uncertainty (p=0.5).
    """
    p = np.clip(float(prob), eps, 1.0 - eps)
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


class VoiceEnsemble:
    """A local ensemble of classifiers with per-member calibrated probabilities,
    predictive entropy, confidence margins, and epistemic model disagreement.
    """

    def __init__(self, random_state: int = 42, feature_columns: Optional[List[str]] = None):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.members = {
            "logreg": LogisticRegression(max_iter=2000, C=1.0, random_state=random_state),
            "rf": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=random_state),
            "gbm": GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=random_state),
        }
        self.calibrated: Dict[str, CalibratedClassifierCV] = {}
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)


    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Validate columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        self.feature_columns = available_cols
        X_imp = self.imputer.fit_transform(X[self.feature_columns])
        Xs = self.scaler.fit_transform(X_imp)
        for name, clf in self.members.items():
            cal = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
            cal.fit(Xs, y)
            self.calibrated[name] = cal
        return self

    def predict_proba_members(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = self.imputer.transform(X[self.feature_columns])
        Xs = self.scaler.transform(X_imp)
        out = {}
        for name, cal in self.calibrated.items():
            out[name] = cal.predict_proba(Xs)[:, 1]
        return pd.DataFrame(out, index=X.index)

    def predict_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return per-row mean probability, std across members (disagreement),
        confidence margin, and Shannon predictive entropy.
        """
        probs = self.predict_proba_members(X)
        mean_p = probs.mean(axis=1)
        std_p = probs.std(axis=1)
        margin = (mean_p - 0.5).abs() * 2.0  # 0 = uncertain, 1 = confident
        entropy = mean_p.apply(compute_binary_entropy)
        
        summary = pd.DataFrame({
            "pd_probability": mean_p,
            "calibrated_probability": mean_p,
            "model_disagreement": std_p,
            "confidence_margin": margin,
            "predictive_entropy": entropy,
            "epistemic_uncertainty": std_p,
        }, index=X.index)
        return pd.concat([summary, probs.add_prefix("prob_")], axis=1)

    def predict_single_detailed(self, row_or_df: Any) -> PredictionResult:
        """Produces a typed PredictionResult schema for a single instance."""
        if isinstance(row_or_df, pd.Series):
            df = pd.DataFrame([row_or_df])
        elif isinstance(row_or_df, pd.DataFrame):
            df = row_or_df
        else:
            df = pd.DataFrame([row_or_df])

        summary = self.predict_summary(df).iloc[0]
        member_probs = {k: float(summary[f"prob_{k}"]) for k in self.members.keys()}

        return PredictionResult(
            pd_probability=float(summary["pd_probability"]),
            calibrated_probability=float(summary["calibrated_probability"]),
            confidence_margin=float(summary["confidence_margin"]),
            model_disagreement=float(summary["model_disagreement"]),
            predictive_entropy=float(summary["predictive_entropy"]),
            epistemic_uncertainty=float(summary["epistemic_uncertainty"]),
            member_probabilities=member_probs,
        )

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(MODEL_DIR, "voice_ensemble.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Optional[str] = None) -> "VoiceEnsemble":
        path = path or os.path.join(MODEL_DIR, "voice_ensemble.joblib")
        return joblib.load(path)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Computes ECE and reliability diagram data bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        count = int(mask.sum())
        bin_counts.append(count)
        center = float((bins[i] + bins[i + 1]) / 2.0)
        bin_centers.append(center)
        if count > 0:
            acc = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            ece += (count / len(y_prob)) * abs(acc - conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(center)

    details = {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }
    return float(ece), details


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """Patient-level split (strictly no patient overlap across train/test),
    train the ensemble, and report discrimination/calibration metrics.
    """
    patients = np.array(df["patient_id"].unique().tolist(), dtype=object)
    train_p, test_p = train_test_split(patients, test_size=test_size, random_state=seed)
    train_df = df[df["patient_id"].isin(train_p)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_p)].reset_index(drop=True)

    ens = VoiceEnsemble(random_state=seed).fit(train_df, train_df["label"])
    pred = ens.predict_summary(test_df)

    auc = float(roc_auc_score(test_df["label"], pred["pd_probability"]))
    brier = float(brier_score_loss(test_df["label"], pred["pd_probability"]))
    ece, cal_details = compute_expected_calibration_error(test_df["label"].values, pred["pd_probability"].values)

    metrics = {
        "auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "n_train_patients": len(train_p),
        "n_test_patients": len(test_p),
        "n_test_visits": len(test_df),
        "calibration_details": cal_details,
    }
    return ens, metrics, train_df, test_df
