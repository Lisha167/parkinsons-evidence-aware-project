"""
baseline.py
---------------------------------------------------------------
Patient-Specific Baseline & Robust Change Detection.

Implements robust individualized baselines using strictly prior visits:
- Robust location estimate: Median
- Robust dispersion estimate: Normalized Median Absolute Deviation (MAD * 1.4826)
- Variance flooring against near-zero historical variability
- Minimum clinical effect size threshold (Delta_min)
- Temporal persistence tracking (distinguishing transient fluctuation from sustained change)
- Multi-axis phenotype agreement verification
"""
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from src.features import PHENOTYPE_COLUMNS
from src.schemas import BaselineResult, ChangeDetectionResult

MIN_VISITS_FOR_BASELINE = 2
Z_ROBUST_THRESHOLD = 1.75  # robust MAD z-score threshold
MIN_EFFECT_THRESHOLD = 0.50  # minimum absolute normalized effect size
MAD_SCALE_FACTOR = 1.4826  # asymptotically normal consistent estimator
MIN_DISPERSION_FLOOR = 1e-3


def compute_patient_baseline(
    patient_history: pd.DataFrame,
    upto_visit_id: Union[int, float],
    feature_columns: Optional[List[str]] = None,
    time_col: str = "visit_id",
) -> Optional[BaselineResult]:
    """Build baseline stats from all visits STRICTLY BEFORE upto_visit_id.
    Never uses current or future visit data (zero leakage guarantee).
    """
    cols_to_use = feature_columns or PHENOTYPE_COLUMNS
    if time_col not in patient_history.columns:
        if "test_time" in patient_history.columns:
            time_col = "test_time"
        else:
            return None

    prior = patient_history[patient_history[time_col] < upto_visit_id]
    if len(prior) < MIN_VISITS_FOR_BASELINE:
        return None

    locations: Dict[str, float] = {}
    dispersions: Dict[str, float] = {}

    for col in cols_to_use:
        if col not in prior.columns:
            continue
        vals = prior[col].dropna().values

        if len(vals) < MIN_VISITS_FOR_BASELINE:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        norm_mad = float(max(mad * MAD_SCALE_FACTOR, MIN_DISPERSION_FLOOR))
        
        locations[col] = round(med, 4)
        dispersions[col] = round(norm_mad, 4)

    return BaselineResult(
        has_baseline=True,
        baseline_n_visits=len(prior),
        baseline_method="robust_median_mad",
        location_estimates=locations,
        dispersion_estimates=dispersions,
        min_effect_threshold=MIN_EFFECT_THRESHOLD,
    )


def evaluate_change(
    current_row: Union[pd.Series, Dict[str, Any]],
    baseline: Optional[BaselineResult],
    patient_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluates whether current observation represents a meaningful, sustained
    patient-specific deviation or an expected within-person fluctuation.
    """
    if baseline is None or not baseline.has_baseline:
        res = ChangeDetectionResult(
            has_baseline=False,
            baseline_n_visits=0,
            axis_deviations={},
            max_abs_deviation=None,
            driving_axis=None,
            effect_size_exceeded=False,
            temporal_persistence_count=0,
            meaningful_change=False,
            verdict="no_baseline_insufficient_history",
            method="robust_mad_persistence",
            details={"reason": "Fewer than 2 prior visits available to construct individualized baseline."},
        )
        return res.to_dict()

    row_dict = current_row.to_dict() if isinstance(current_row, pd.Series) else current_row
    deviations: Dict[str, float] = {}
    effect_sizes: Dict[str, float] = {}

    cols_to_evaluate = list(baseline.location_estimates.keys())
    for col in cols_to_evaluate:
        if col not in row_dict:
            continue
        val = row_dict[col]
        if pd.isna(val):
            continue

        med = baseline.location_estimates[col]
        disp = baseline.dispersion_estimates[col]
        z_mad = float((val - med) / disp)
        deviations[col] = round(z_mad, 3)
        effect_sizes[col] = round(abs(val - med), 3)

    if not deviations:
        res = ChangeDetectionResult(
            has_baseline=True,
            baseline_n_visits=baseline.baseline_n_visits,
            axis_deviations={},
            max_abs_deviation=None,
            driving_axis=None,
            effect_size_exceeded=False,
            temporal_persistence_count=0,
            meaningful_change=False,
            verdict="insufficient_feature_data",
        )
        return res.to_dict()

    driving_axis = max(deviations, key=lambda k: abs(deviations[k]))
    max_abs_z = float(abs(deviations[driving_axis]))
    
    # 1. Statistical deviation threshold check
    stat_exceeded = max_abs_z >= Z_ROBUST_THRESHOLD
    
    # 2. Minimum clinical effect threshold
    effect_exceeded = any(effect_sizes.get(c, 0.0) >= MIN_EFFECT_THRESHOLD * baseline.dispersion_estimates.get(c, 1.0) for c in deviations)
    
    # 3. Multi-axis agreement (do at least 2 axes show positive Parkinsonian shift?)
    positive_shifts = sum(1 for v in deviations.values() if v >= 1.0)
    multi_axis_agreement = positive_shifts >= 2 or max_abs_z >= 2.5

    # 4. Temporal persistence check
    # Check if the previous visit (if available) also showed deviation
    persistence_count = 1
    current_vid = int(row_dict.get("visit_id", 0))
    if patient_history is not None and current_vid > 2:
        prev_prior = patient_history[patient_history["visit_id"] < current_vid]
        if len(prev_prior) >= 2:
            prev_visit = prev_prior.sort_values("visit_id").iloc[-1]
            prev_baseline = compute_patient_baseline(patient_history, prev_visit["visit_id"])
            if prev_baseline is not None and driving_axis in prev_baseline.location_estimates:
                p_med = prev_baseline.location_estimates[driving_axis]
                p_disp = prev_baseline.dispersion_estimates[driving_axis]
                p_z = (prev_visit[driving_axis] - p_med) / p_disp
                if abs(p_z) >= 1.2:
                    persistence_count = 2

    # Sustained meaningful change decision rule
    is_meaningful = bool((stat_exceeded and effect_exceeded and multi_axis_agreement) or (stat_exceeded and persistence_count >= 2))

    if is_meaningful:
        verdict = "meaningful_patient_specific_change"
    else:
        verdict = "within_expected_fluctuation"

    res = ChangeDetectionResult(
        has_baseline=True,
        baseline_n_visits=baseline.baseline_n_visits,
        axis_deviations=deviations,
        max_abs_deviation=max_abs_z,
        driving_axis=driving_axis,
        effect_size_exceeded=effect_exceeded,
        temporal_persistence_count=persistence_count,
        meaningful_change=is_meaningful,
        verdict=verdict,
        method="robust_mad_persistence",
        details={
            "multi_axis_positive_shifts": positive_shifts,
            "stat_exceeded": stat_exceeded,
            "effect_exceeded": effect_exceeded,
            "persistence_count": persistence_count,
        },
    )
    return res.to_dict()


# ---------------------------------------------------------------------
# Benchmark Change Detection Baselines (Section 10)
# ---------------------------------------------------------------------
def evaluate_naive_population_baseline(current_row: Union[pd.Series, Dict[str, Any]], pop_threshold: float = 0.60) -> Dict[str, Any]:
    """A. Naive population threshold baseline."""
    row_dict = current_row.to_dict() if isinstance(current_row, pd.Series) else current_row
    score = float(row_dict.get("phenotype_overall_phenotype_severity", 0.5))
    detected = score >= pop_threshold
    return {
        "method": "naive_population_threshold",
        "score": score,
        "threshold": pop_threshold,
        "meaningful_change": detected,
        "verdict": "meaningful_patient_specific_change" if detected else "within_expected_fluctuation",
    }


def evaluate_simple_zscore_baseline(current_row: Union[pd.Series, Dict[str, Any]], patient_history: pd.DataFrame, upto_visit_id: int, z_thresh: float = 1.5) -> Dict[str, Any]:
    """B. Simple sample-mean/std patient z-score baseline."""
    prior = patient_history[patient_history["visit_id"] < upto_visit_id]
    if len(prior) < MIN_VISITS_FOR_BASELINE:
        return {"method": "simple_zscore", "has_baseline": False, "meaningful_change": False, "verdict": "no_baseline_insufficient_history"}
    
    row_dict = current_row.to_dict() if isinstance(current_row, pd.Series) else current_row
    z_scores = {}
    for col in PHENOTYPE_COLUMNS:
        vals = prior[col].dropna()
        m = float(vals.mean())
        s = float(max(vals.std(ddof=0), 1e-3))
        z_scores[col] = float((row_dict[col] - m) / s) if col in row_dict else 0.0

    max_z = max(abs(v) for v in z_scores.values())
    detected = max_z >= z_thresh
    return {
        "method": "simple_zscore",
        "has_baseline": True,
        "max_abs_z": max_z,
        "meaningful_change": detected,
        "verdict": "meaningful_patient_specific_change" if detected else "within_expected_fluctuation",
    }
