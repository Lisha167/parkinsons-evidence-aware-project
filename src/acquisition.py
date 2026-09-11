"""
acquisition.py
---------------------------------------------------------------
Information-Acquisition Policy & Probabilistic Observation Models.

Implements formal Expected Information Gain (EIG) and Cost/Burden-aware
Utility for selecting the next-best assessment from a predefined feasible space.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.schemas import (
    AssessmentDefinition,
    AssessmentResult,
    AcquisitionDecision,
    EvidenceState,
)
from src.models import compute_binary_entropy

LAMBDA_COST_DEFAULT = 0.50
LAMBDA_BURDEN_DEFAULT = 0.30

# Predefined feasible assessment catalog (Section 4)
ASSESSMENT_CATALOG: Dict[str, AssessmentDefinition] = {
    "repeat_sustained_vowel": AssessmentDefinition(
        assessment_id="repeat_sustained_vowel",
        name="Repeat Sustained Vowel",
        evidence_type="acoustic_phonation",
        cost=0.05,
        burden=0.05,
        expected_duration_minutes=2.0,
        availability="immediate",
        target_gaps={"low_signal_quality": 0.90, "high_predictive_entropy": 0.30},
        description="Re-record sustained /a/ vowel phonation to resolve transient recording artifacts and improve SNR.",
    ),
    "reading_passage_task": AssessmentDefinition(
        assessment_id="reading_passage_task",
        name="Standardized Reading Passage",
        evidence_type="acoustic_continuous_speech",
        cost=0.10,
        burden=0.10,
        expected_duration_minutes=4.0,
        availability="immediate",
        target_gaps={"low_predictive_confidence": 0.65, "phenotype_mismatch": 0.70, "high_predictive_entropy": 0.60},
        description="Standardized 'Rainbow Passage' or 'Grandfather Passage' evaluating prosodic variation and speech rate.",
    ),
    "diadochokinetic_task": AssessmentDefinition(
        assessment_id="diadochokinetic_task",
        name="Diadochokinetic (DDK) Speech Task",
        evidence_type="acoustic_articulatory",
        cost=0.15,
        burden=0.15,
        expected_duration_minutes=5.0,
        availability="immediate",
        target_gaps={"high_model_disagreement": 0.75, "low_predictive_confidence": 0.70, "high_predictive_entropy": 0.70},
        description="Rapid syllable repetition (/pa-ta-ka/) capturing articulatory precision and motor speech rhythmicity.",
    ),
    "extended_home_monitoring": AssessmentDefinition(
        assessment_id="extended_home_monitoring",
        name="7-Day Longitudinal Home Voice Diary",
        evidence_type="longitudinal_ecological",
        cost=0.30,
        burden=0.45,
        expected_duration_minutes=10080.0,
        availability="delayed_1_week",
        target_gaps={"unestablished_baseline": 0.85, "temporal_predictive_tension": 0.80},
        description="Passive smartphone voice logging over 7 days to establish intra-individual variability norms.",
    ),
    "accelerometer_gait_check": AssessmentDefinition(
        assessment_id="accelerometer_gait_check",
        name="Wearable Inertial Gait & Tremor Check",
        evidence_type="multimodal_inertial",
        cost=0.35,
        burden=0.35,
        expected_duration_minutes=15.0,
        availability="clinic_or_sensor",
        target_gaps={"phenotype_mismatch": 0.80, "high_model_disagreement": 0.60, "low_predictive_confidence": 0.65},
        description="Cross-modality wearable sensor test measuring postural tremor and gait stride variability.",
    ),
    "clinical_updrs_exam": AssessmentDefinition(
        assessment_id="clinical_updrs_exam",
        name="Clinician UPDRS-III Motor Examination",
        evidence_type="clinical_gold_standard",
        cost=0.90,
        burden=0.70,
        expected_duration_minutes=45.0,
        availability="specialist_appointment",
        target_gaps={
            "low_signal_quality": 0.50,
            "low_predictive_confidence": 0.95,
            "high_model_disagreement": 0.95,
            "high_predictive_entropy": 0.95,
            "phenotype_mismatch": 0.95,
            "temporal_predictive_tension": 0.90,
        },
        description="In-person neurological examination by a movement disorder specialist (MDS-UPDRS Part III).",
    ),
}


# ---------------------------------------------------------------------
# Probabilistic Assessment Observation Models (Section 6)
# ---------------------------------------------------------------------
def sample_assessment_observation(
    current_evidence_row: pd.Series,
    assessment_id: str,
    rng: Optional[np.random.Generator] = None,
) -> AssessmentResult:
    """Samples probabilistic task-specific observations conditioned on the
    underlying latent patient state.
    """
    rng = rng or np.random.default_rng()
    defn = ASSESSMENT_CATALOG[assessment_id]
    row = current_evidence_row.copy()
    is_pd = bool(row.get("label", 1 if row.get("pd_probability", 0.5) >= 0.5 else 0))
    current_sq = float(row.get("signal_quality", 0.8))

    obs_data: Dict[str, Any] = {}
    quality_gain = 0.0
    obs_noise = float(rng.normal(0.0, 0.03))

    if assessment_id == "repeat_sustained_vowel":
        # Resolves acoustic recording noise
        quality_gain = float(rng.uniform(0.18, 0.35))
        new_sq = float(np.clip(current_sq + quality_gain, 0.10, 1.0))
        # Reduce acoustic measurement jitter noise
        jitter_clean = float(max(0.001, row.get("jitter_pct", 0.005) * rng.uniform(0.88, 1.02)))
        shimmer_clean = float(max(0.01, row.get("shimmer", 0.03) * rng.uniform(0.88, 1.02)))
        obs_data = {"new_signal_quality": new_sq, "jitter_pct": jitter_clean, "shimmer": shimmer_clean}

    elif assessment_id == "reading_passage_task":
        # Captures pitch variability (spread1/spread2) and prosodic continuity
        prosodic_drop = float(rng.normal(0.12 * is_pd, 0.03))
        obs_data = {
            "prosodic_variability": float(max(0.0, 0.50 - prosodic_drop)),
            "speech_rate_syllables_per_sec": float(rng.normal(4.2 - 1.0 * is_pd, 0.4)),
            "spread1": float(row.get("spread1", -5.0) + (1.2 if is_pd else -0.5)),
            "new_signal_quality": float(np.clip(current_sq + 0.08, 0.10, 1.0)),
        }

    elif assessment_id == "diadochokinetic_task":
        # Captures syllable repetition regularity and precision
        ddk_rate = float(rng.normal(5.8 - 1.6 * is_pd, 0.5))
        ddk_jitter = float(rng.normal(0.02 + 0.06 * is_pd, 0.01))
        obs_data = {
            "ddk_syllable_rate": ddk_rate,
            "ddk_regularity_index": float(np.clip(1.0 - ddk_jitter * 8.0, 0.0, 1.0)),
            "rpde": float(np.clip(row.get("rpde", 0.45) + (0.10 if is_pd else -0.05), 0.1, 0.95)),
            "ppe": float(np.clip(row.get("ppe", 0.20) + (0.12 if is_pd else -0.05), 0.01, 0.98)),
        }

    elif assessment_id == "extended_home_monitoring":
        # 7-day longitudinal sampling reduces within-person variance and verifies baseline
        obs_data = {
            "7day_mean_f0_stability": float(rng.normal(0.85 - 0.25 * is_pd, 0.05)),
            "confirmed_baseline_variance": float(rng.uniform(0.02, 0.05)),
            "diurnal_fluctuation_present": bool(is_pd and rng.random() > 0.3),
        }

    elif assessment_id == "accelerometer_gait_check":
        # Wearable motor sensor data
        tremor_power = float(rng.exponential(2.5 * is_pd + 0.2))
        stride_cv = float(rng.normal(0.03 + 0.05 * is_pd, 0.01))
        obs_data = {
            "resting_tremor_power_hz": tremor_power,
            "stride_time_coefficient_variation": stride_cv,
            "multimodal_pd_concordance": float(np.clip(0.3 + 0.6 * is_pd, 0.0, 1.0)),
        }

    elif assessment_id == "clinical_updrs_exam":
        # Neurologist clinical examination
        updrs_score = float(max(0.0, rng.normal(26.0 * is_pd + 3.0, 3.5)))
        obs_data = {
            "updrs_part_iii_score": updrs_score,
            "bradykinesia_subscore": float(np.clip(updrs_score / 12.0, 0.0, 4.0)),
            "rigidity_subscore": float(np.clip(updrs_score / 15.0, 0.0, 4.0)),
            "clinical_concurrence": bool(is_pd if updrs_score >= 15.0 else not is_pd),
        }

    return AssessmentResult(
        assessment_id=defn.assessment_id,
        name=defn.name,
        cost=defn.cost,
        burden=defn.burden,
        evidence_data=obs_data,
        observation_noise=round(obs_noise, 4),
        quality_gain=round(quality_gain, 4),
    )


# ---------------------------------------------------------------------
# Formal Next-Best-Assessment Policy Formulation (Section 5)
# ---------------------------------------------------------------------
def estimate_expected_information_gain(
    candidate: AssessmentDefinition,
    current_prob: float,
    current_entropy: float,
    evidence_gaps: List[str],
) -> Tuple[float, float]:
    """Calculates formal Expected Information Gain:
    EIG(a) = H(D | E) - E_{e_a}[ H(D | E, e_a) ]
    and Expected Decision Loss Reduction:
    Delta L(a) = L(E) - E_{e_a}[ L(E, e_a) ]
    where L(p) = min(p, 1-p).
    """
    # 1. Base potential reduction power of this assessment for the current gaps
    gap_weights = [candidate.target_gaps.get(g, 0.10) for g in evidence_gaps]
    relevance = float(np.mean(gap_weights)) if gap_weights else 0.20

    # 2. Model the predictive distribution of posterior probability P(D | E, e_a)
    # An informative test shifts the posterior away from 0.5 toward either 0 or 1
    # with variance proportional to relevance * entropy
    shift_magnitude = float(0.40 * relevance * current_entropy)
    
    # Probabilistic outcomes of test: positive vs negative findings
    p_pos = current_prob
    p_neg = 1.0 - current_prob

    post_p_if_pos = float(np.clip(current_prob + shift_magnitude * (1.0 - current_prob), 0.01, 0.99))
    post_p_if_neg = float(np.clip(current_prob - shift_magnitude * current_prob, 0.01, 0.99))

    post_entropy_pos = compute_binary_entropy(post_p_if_pos)
    post_entropy_neg = compute_binary_entropy(post_p_if_neg)

    expected_post_entropy = p_pos * post_entropy_pos + p_neg * post_entropy_neg
    eig = float(max(0.0, current_entropy - expected_post_entropy))

    # Decision loss reduction (0-1 risk)
    current_loss = min(current_prob, 1.0 - current_prob)
    post_loss_pos = min(post_p_if_pos, 1.0 - post_p_if_pos)
    post_loss_neg = min(post_p_if_neg, 1.0 - post_p_if_neg)
    expected_post_loss = p_pos * post_loss_pos + p_neg * post_loss_neg
    loss_reduction = float(max(0.0, current_loss - expected_post_loss))

    return round(eig, 4), round(loss_reduction, 4)


def compute_formal_utility(
    eig: float,
    loss_reduction: float,
    cost: float,
    burden: float,
    lambda_cost: float = LAMBDA_COST_DEFAULT,
    lambda_burden: float = LAMBDA_BURDEN_DEFAULT,
) -> float:
    """Utility(a | E) = EIG(a | E) - lambda_cost * Cost(a) - lambda_burden * Burden(a)."""
    utility = eig - (lambda_cost * cost + lambda_burden * burden)
    return round(float(utility), 4)


def recommend_next_assessment(
    evidence: Dict[str, Any],
    pred_summary: Optional[Dict[str, Any]] = None,
    lambda_cost: float = LAMBDA_COST_DEFAULT,
    lambda_burden: float = LAMBDA_BURDEN_DEFAULT,
    excluded_assessments: Optional[List[str]] = None,
    dataset_id: Optional[str] = None,
) -> AcquisitionDecision:
    """Computes the optimal next assessment using formal EIG and cost/burden penalties.
    If dataset_id is provided, filters out assessments unsupported by that dataset.
    """
    from src.data.assessment_capability import is_assessment_available

    excluded = set(excluded_assessments or [])
    current_prob = float(pred_summary.get("pd_probability", 0.5) if pred_summary else 0.5)
    current_entropy = float(pred_summary.get("predictive_entropy", compute_binary_entropy(current_prob)) if pred_summary else compute_binary_entropy(current_prob))
    
    dominant_gaps = evidence.get("dominant_gaps", [])
    if not dominant_gaps:
        if evidence.get("signal_quality", 1.0) < 0.55:
            dominant_gaps.append("low_signal_quality")
        if evidence.get("confidence_margin", 1.0) < 0.35:
            dominant_gaps.append("low_predictive_confidence")
        if evidence.get("model_disagreement", 0.0) > 0.12:
            dominant_gaps.append("high_model_disagreement")

    candidates_scored = []
    for a_id, defn in ASSESSMENT_CATALOG.items():
        if a_id in excluded:
            continue
        if dataset_id and not is_assessment_available(a_id, dataset_id):
            continue

        eig, loss_red = estimate_expected_information_gain(defn, current_prob, current_entropy, dominant_gaps)
        util = compute_formal_utility(eig, loss_red, defn.cost, defn.burden, lambda_cost, lambda_burden)
        candidates_scored.append({
            "assessment_id": a_id,
            "name": defn.name,
            "expected_info_gain": eig,
            "expected_loss_reduction": loss_red,
            "cost": defn.cost,
            "burden": defn.burden,
            "utility": util,
            "description": defn.description,
        })

    candidates_scored.sort(key=lambda x: x["utility"], reverse=True)
    best = candidates_scored[0]

    rationale = [
        f"Selected '{best['name']}' with formal Expected Info Gain (EIG) = {best['expected_info_gain']:.3f} bits.",
        f"Considers cost penalty ({best['cost']:.2f}) and burden penalty ({best['burden']:.2f}) yielding net utility = {best['utility']:.3f}.",
        f"Directly addresses evidentiary gaps: {', '.join(dominant_gaps) if dominant_gaps else 'baseline validation'}.",
    ]

    return AcquisitionDecision(
        assessment=best["assessment_id"],
        expected_info_gain=best["expected_info_gain"],
        expected_loss_reduction=best["expected_loss_reduction"],
        cost=best["cost"],
        burden=best["burden"],
        utility=best["utility"],
        policy_name="cost_burden_aware_eig",
        description=best["description"],
        rationale=rationale,
        candidates_ranked=candidates_scored,
    )
