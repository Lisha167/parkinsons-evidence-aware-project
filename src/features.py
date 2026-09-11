"""
features.py
---------------------------------------------------------------
Defines the acoustic feature schema, signal quality assessment,
and builds the "structured speech phenotype": a compact, clinically-readable
summary of vocal stability, intensity/amplitude variation, noise
characteristics, and nonlinear dynamics.
"""
from typing import Dict, List, Any, Optional, Protocol
import numpy as np
import pandas as pd

from src.schemas import SignalQualityMetrics, PhenotypeResult

from dataclasses import dataclass, field

RAW_FEATURES = [
    "mdvp_fo", "mdvp_fhi", "mdvp_flo",
    "jitter_pct", "jitter_abs", "rap", "ppq", "ddp",
    "shimmer", "shimmer_db", "apq3", "apq5", "apq11", "dda",
    "nhr", "hnr", "rpde", "dfa", "ppe", "spread1", "spread2", "d2",
]

# Grouping raw acoustic measures into interpretable phenotype axes
PHENOTYPE_GROUPS = {
    "vocal_stability": ["jitter_pct", "jitter_abs", "rap", "ppq", "ddp", "rpde"],
    "amplitude_variation": ["shimmer", "shimmer_db", "apq3", "apq5", "apq11", "dda"],
    "noise_characteristics": ["nhr", "hnr"],
    "nonlinear_dynamics": ["dfa", "ppe", "spread1", "spread2", "d2"],
}

# Reference ranges approximate healthy-population norms (synthetic & real data compatible)
REFERENCE_RANGES = {
    "mdvp_fo": (80.0, 260.0), "mdvp_fhi": (100.0, 400.0), "mdvp_flo": (60.0, 220.0),
    "jitter_pct": (0.001, 0.02), "jitter_abs": (0.00001, 0.0003),
    "rap": (0.0005, 0.01), "ppq": (0.0005, 0.01), "ddp": (0.0015, 0.03),
    "rpde": (0.25, 0.7),
    "shimmer": (0.01, 0.09), "shimmer_db": (0.1, 1.0),
    "apq3": (0.005, 0.045), "apq5": (0.006, 0.05), "apq11": (0.008, 0.07),
    "dda": (0.015, 0.14),
    "nhr": (0.005, 0.08), "hnr": (28.0, 12.0),  # inverted: lower HNR = worse
    "dfa": (0.72, 0.55),  # inverted: lower DFA = worse
    "ppe": (0.05, 0.5), "spread1": (-8.0, -3.0), "spread2": (0.1, 0.35), "d2": (2.0, 3.2),
}


@dataclass
class FeatureSchema:
    """Configurable acoustic feature schema representation."""
    schema_name: str = "UCI_22_MDVP"
    raw_features: List[str] = field(default_factory=lambda: list(RAW_FEATURES))
    phenotype_groups: Dict[str, List[str]] = field(default_factory=lambda: dict(PHENOTYPE_GROUPS))
    reference_ranges: Dict[str, Any] = field(default_factory=lambda: dict(REFERENCE_RANGES))

    @classmethod
    def default(cls) -> "FeatureSchema":
        return cls()

    @property
    def phenotype_columns(self) -> List[str]:
        return [f"phenotype_{k}" for k in list(self.phenotype_groups.keys()) + ["overall_phenotype_severity"]]



def compute_signal_quality_metrics(row: pd.Series) -> SignalQualityMetrics:
    """Computes comprehensive signal quality indicators including missingness,
    outliers, boundary plausibility, SNR proxy, and overall confidence in the recording.
    """
    # 1. Missingness detection
    missing_count = sum(pd.isna(row.get(f, np.nan)) for f in RAW_FEATURES)
    missingness_rate = missing_count / len(RAW_FEATURES)

    # 2. Outlier and boundary plausibility
    outlier_violations = 0
    for f in RAW_FEATURES:
        val = row.get(f, None)
        if val is None or pd.isna(val):
            continue
        lo, hi = REFERENCE_RANGES[f]
        min_b = min(lo, hi) - 2.0 * abs(hi - lo)
        max_b = max(lo, hi) + 2.0 * abs(hi - lo)
        if val < min_b or val > max_b:
            outlier_violations += 1
    outlier_score = min(1.0, outlier_violations / max(1, len(RAW_FEATURES)))

    # 3. SNR / Harmonic Quality proxy (derived from HNR if available)
    hnr_val = row.get("hnr", 20.0)
    snr_proxy = float(np.clip((hnr_val - 5.0) / 25.0, 0.0, 1.0)) if not pd.isna(hnr_val) else 0.5

    # 4. Boundary validity
    boundary_validity = float(max(0.0, 1.0 - (missingness_rate * 0.7 + outlier_score * 0.3)))

    # Raw recording signal quality if present in input row
    raw_sq = float(row.get("signal_quality", 0.85)) if not pd.isna(row.get("signal_quality", np.nan)) else 0.85

    # Combined overall quality
    overall = float(np.clip(0.5 * raw_sq + 0.3 * snr_proxy + 0.2 * boundary_validity - 0.4 * missingness_rate, 0.0, 1.0))
    quality_uncertainty = float(1.0 - overall)
    is_acceptable = bool(overall >= 0.55 and missingness_rate < 0.2 and outlier_score < 0.3)

    return SignalQualityMetrics(
        overall_quality=round(overall, 3),
        missingness_rate=round(missingness_rate, 3),
        outlier_score=round(outlier_score, 3),
        snr_proxy=round(snr_proxy, 3),
        boundary_validity=round(boundary_validity, 3),
        quality_uncertainty=round(quality_uncertainty, 3),
        is_acceptable=is_acceptable,
    )


def build_phenotype(row: pd.Series) -> dict:
    """Collapse raw acoustic features into normalized phenotype axis scores
    (0-1, higher = more Parkinsonian-like on that axis) using transparent
    reference-range normalization.
    """
    def norm(name: str) -> float:
        lo, hi = REFERENCE_RANGES[name]
        val = row.get(name, lo)
        if pd.isna(val):
            return 0.5
        score = (val - lo) / (hi - lo)
        return float(np.clip(score, 0.0, 1.0))

    axis_scores = {}
    for axis, members in PHENOTYPE_GROUPS.items():
        axis_scores[axis] = float(np.mean([norm(m) for m in members]))

    axis_scores["overall_phenotype_severity"] = float(np.mean(list(axis_scores.values())))
    return axis_scores


def get_structured_phenotype(row: pd.Series) -> PhenotypeResult:
    scores = build_phenotype(row)
    return PhenotypeResult(
        vocal_stability=scores["vocal_stability"],
        amplitude_variation=scores["amplitude_variation"],
        noise_characteristics=scores["noise_characteristics"],
        nonlinear_dynamics=scores["nonlinear_dynamics"],
        overall_phenotype_severity=scores["overall_phenotype_severity"],
        raw_axis_scores=scores,
    )


def add_phenotype_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    phen = df.apply(build_phenotype, axis=1, result_type="expand")
    phen.columns = [f"phenotype_{c}" for c in phen.columns]
    return pd.concat([df, phen], axis=1)


PHENOTYPE_COLUMNS = [f"phenotype_{k}" for k in list(PHENOTYPE_GROUPS.keys()) + ["overall_phenotype_severity"]]


# ---------------------------------------------------------------------
# Clean Interface Protocols for Real-Data Compatibility (Section 28)
# ---------------------------------------------------------------------
class DataLoader(Protocol):
    def load_dataset(self, source_path: str) -> pd.DataFrame:
        ...


class FeatureExtractor(Protocol):
    def extract_features(self, raw_audio_or_dict: Any) -> Dict[str, float]:
        """Interface for raw-audio or pre-extracted acoustic feature extraction."""
        ...


class PatientTimeline(Protocol):
    def get_patient_history(self, patient_id: str, up_to_visit_id: Optional[int] = None) -> pd.DataFrame:
        ...


class AssessmentProvider(Protocol):
    def acquire_assessment(self, patient_id: str, assessment_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ...
