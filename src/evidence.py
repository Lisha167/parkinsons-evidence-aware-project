"""
evidence.py
---------------------------------------------------------------
Evidence-Assessment Engine & Evidence Gap Identification.

Determines the computational evidentiary state (SUFFICIENT, INSUFFICIENT,
CONTRADICTORY) and explicitly extracts dominant evidence gaps and uncertainty sources.
"""
from typing import Dict, List, Any, Optional
import numpy as np

from src.schemas import EvidenceState, EvidenceGap, SignalQualityMetrics

THRESH = dict(
    min_signal_quality=0.55,
    min_confidence_margin=0.35,      # |p-0.5|*2 below this = low confidence
    max_model_disagreement=0.12,     # std across ensemble members
    max_predictive_entropy=0.88,     # binary Shannon entropy in bits (max 1.0)
    min_phenotype_consistency=0.60,  # agreement between rule phenotype & ML prediction
)


def phenotype_consistency_score(pred_summary: Dict[str, Any], phenotype: Dict[str, Any]) -> float:
    """Calculates concordance between rule-based phenotypic severity and ML disease probability."""
    phen_severity = float(phenotype.get("phenotype_overall_phenotype_severity",
                          phenotype.get("overall_phenotype_severity", 0.5)))
    pd_prob = float(pred_summary.get("pd_probability", 0.5))
    return float(np.clip(1.0 - abs(phen_severity - pd_prob), 0.0, 1.0))


def identify_evidence_gaps(
    signal_quality: float,
    confidence_margin: float,
    model_disagreement: float,
    predictive_entropy: float,
    phenotype_consistency: float,
    temporal_status: str,
) -> List[EvidenceGap]:
    """Explicitly extracts and quantifies specific evidentiary gaps."""
    gaps: List[EvidenceGap] = []

    if signal_quality < THRESH["min_signal_quality"]:
        sev = min(1.0, (THRESH["min_signal_quality"] - signal_quality) / THRESH["min_signal_quality"] + 0.3)
        gaps.append(EvidenceGap("low_signal_quality", round(sev, 3), f"Signal quality ({signal_quality:.2f}) below threshold."))

    if confidence_margin < THRESH["min_confidence_margin"]:
        sev = min(1.0, (THRESH["min_confidence_margin"] - confidence_margin) / THRESH["min_confidence_margin"] + 0.2)
        gaps.append(EvidenceGap("low_predictive_confidence", round(sev, 3), f"Confidence margin ({confidence_margin:.2f}) indicates decision boundary ambiguity."))

    if model_disagreement > THRESH["max_model_disagreement"]:
        sev = min(1.0, (model_disagreement - THRESH["max_model_disagreement"]) / 0.15 + 0.3)
        gaps.append(EvidenceGap("high_model_disagreement", round(sev, 3), f"Ensemble models disagree significantly (std={model_disagreement:.2f})."))

    if predictive_entropy > THRESH["max_predictive_entropy"]:
        sev = min(1.0, (predictive_entropy - THRESH["max_predictive_entropy"]) / 0.2 + 0.2)
        gaps.append(EvidenceGap("high_predictive_entropy", round(sev, 3), f"Predictive Shannon entropy ({predictive_entropy:.2f} bits) is elevated."))

    if phenotype_consistency < THRESH["min_phenotype_consistency"]:
        sev = min(1.0, (THRESH["min_phenotype_consistency"] - phenotype_consistency) / 0.4 + 0.3)
        gaps.append(EvidenceGap("phenotype_mismatch", round(sev, 3), f"Acoustic phenotype differs from ML probability (concordance={phenotype_consistency:.2f})."))

    if temporal_status == "no_baseline_insufficient_history":
        gaps.append(EvidenceGap("unestablished_baseline", 0.50, "No longitudinal baseline available to validate patient-specific trajectory."))
    elif temporal_status == "meaningful_patient_specific_change" and confidence_margin < 0.50:
        gaps.append(EvidenceGap("temporal_predictive_tension", 0.70, "Detected patient-specific change is not matched by definitive predictive confidence."))

    return sorted(gaps, key=lambda g: g.severity, reverse=True)


def assess_evidence(
    pred_summary: Dict[str, Any],
    phenotype: Dict[str, Any],
    change_report: Dict[str, Any],
    signal_metrics: Optional[SignalQualityMetrics] = None,
) -> EvidenceState:
    """Computationally evaluates evidence sufficiency, extracting dominant gaps
    and assigning a definitive evidentiary state.
    """
    if signal_metrics is not None:
        signal_quality = signal_metrics.overall_quality
    else:
        signal_quality = float(phenotype.get("signal_quality", pred_summary.get("signal_quality", 1.0)))

    confidence_margin = float(pred_summary.get("confidence_margin", 0.5))
    model_disagreement = float(pred_summary.get("model_disagreement", 0.0))
    predictive_entropy = float(pred_summary.get("predictive_entropy", 0.5))
    phen_consistency = phenotype_consistency_score(pred_summary, phenotype)
    temporal_status = str(change_report.get("verdict", "no_baseline_insufficient_history"))

    reasons_insufficient: List[str] = []
    uncertainty_sources: List[str] = []

    if signal_quality < THRESH["min_signal_quality"]:
        msg = f"Recording signal quality ({signal_quality:.2f}) below threshold ({THRESH['min_signal_quality']:.2f})."
        reasons_insufficient.append(msg)
        uncertainty_sources.append("acoustic_noise_or_artifact")

    if confidence_margin < THRESH["min_confidence_margin"]:
        msg = f"Predictive confidence margin ({confidence_margin:.2f}) too low (< {THRESH['min_confidence_margin']:.2f})."
        reasons_insufficient.append(msg)
        uncertainty_sources.append("model_boundary_uncertainty")

    if model_disagreement > THRESH["max_model_disagreement"]:
        msg = f"Model disagreement ({model_disagreement:.2f}) exceeds threshold ({THRESH['max_model_disagreement']:.2f})."
        reasons_insufficient.append(msg)
        uncertainty_sources.append("epistemic_model_variance")

    if predictive_entropy > THRESH["max_predictive_entropy"]:
        msg = f"Predictive entropy ({predictive_entropy:.2f} bits) exceeds maximum certainty threshold."
        reasons_insufficient.append(msg)
        uncertainty_sources.append("information_entropy")

    reasons_contradictory: List[str] = []
    if phen_consistency < THRESH["min_phenotype_consistency"]:
        reasons_contradictory.append(
            f"Speech phenotype severity is discordant with ML prediction (consistency={phen_consistency:.2f})."
        )
    if temporal_status == "meaningful_patient_specific_change" and confidence_margin < 0.40:
        reasons_contradictory.append(
            "A meaningful patient-specific longitudinal change was detected, but cross-sectional predictive confidence is ambiguous."
        )

    # Determine state
    if reasons_contradictory:
        state = "contradictory"
    elif reasons_insufficient:
        state = "insufficient"
    else:
        state = "sufficient"

    gaps = identify_evidence_gaps(
        signal_quality=signal_quality,
        confidence_margin=confidence_margin,
        model_disagreement=model_disagreement,
        predictive_entropy=predictive_entropy,
        phenotype_consistency=phen_consistency,
        temporal_status=temporal_status,
    )
    dominant_gaps = [g.gap_name for g in gaps if g.severity >= 0.35]

    return EvidenceState(
        decision_state=state,
        signal_quality=round(signal_quality, 3),
        predictive_entropy=round(predictive_entropy, 3),
        confidence_margin=round(confidence_margin, 3),
        model_disagreement=round(model_disagreement, 3),
        phenotype_consistency=round(phen_consistency, 3),
        temporal_status=temporal_status,
        dominant_gaps=dominant_gaps,
        uncertainty_sources=list(set(uncertainty_sources)),
        reasons_insufficient=reasons_insufficient,
        reasons_contradictory=reasons_contradictory,
    )
