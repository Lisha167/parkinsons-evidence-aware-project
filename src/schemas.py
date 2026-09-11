"""
schemas.py
---------------------------------------------------------------
Typed schemas and data contracts for the Evidence-Aware Sequential
Decision-Support System for Parkinson's Voice Assessment.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class PredictionResult:
    pd_probability: float
    calibrated_probability: float
    confidence_margin: float
    model_disagreement: float
    predictive_entropy: float
    epistemic_uncertainty: float
    member_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalQualityMetrics:
    overall_quality: float
    missingness_rate: float
    outlier_score: float
    snr_proxy: float
    boundary_validity: float
    quality_uncertainty: float
    is_acceptable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhenotypeResult:
    vocal_stability: float
    amplitude_variation: float
    noise_characteristics: float
    nonlinear_dynamics: float
    overall_phenotype_severity: float
    raw_axis_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineResult:
    has_baseline: bool
    baseline_n_visits: int
    baseline_method: str
    location_estimates: Dict[str, float] = field(default_factory=dict)
    dispersion_estimates: Dict[str, float] = field(default_factory=dict)
    min_effect_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeDetectionResult:
    has_baseline: bool
    baseline_n_visits: int
    axis_deviations: Dict[str, float]
    max_abs_deviation: Optional[float]
    driving_axis: Optional[str]
    effect_size_exceeded: bool
    temporal_persistence_count: int
    meaningful_change: bool
    verdict: str
    method: str = "robust_mad_persistence"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGap:
    gap_name: str
    severity: float  # 0.0 (no gap) to 1.0 (severe gap)
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceProvenance:
    dataset_id: str
    subject_id: str
    recording_id: Optional[str] = None
    session_id: Optional[str] = None
    assessment_id: Optional[str] = None
    feature_source: str = "acoustic_features"
    model_version: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceState:
    decision_state: str  # 'sufficient', 'insufficient', 'contradictory'
    signal_quality: float
    predictive_entropy: float
    confidence_margin: float
    model_disagreement: float
    phenotype_consistency: float
    temporal_status: str
    dominant_gaps: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    reasons_insufficient: List[str] = field(default_factory=list)
    reasons_contradictory: List[str] = field(default_factory=list)
    provenance: Optional[EvidenceProvenance] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class AssessmentDefinition:
    assessment_id: str
    name: str
    evidence_type: str
    cost: float
    burden: float
    expected_duration_minutes: float
    availability: str
    target_gaps: Dict[str, float]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssessmentResult:
    assessment_id: str
    name: str
    cost: float
    burden: float
    evidence_data: Dict[str, Any]
    observation_noise: float
    quality_gain: float
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AcquisitionDecision:
    assessment: str
    expected_info_gain: float
    expected_loss_reduction: float
    cost: float
    burden: float
    utility: float
    policy_name: str
    description: str
    rationale: List[str] = field(default_factory=list)
    candidates_ranked: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    verifier_id: str
    passed: bool
    verdict: str  # 'acceptable', 'concern', 'rejected'
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    recommended_action: str = ""
    explanation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdversarialChallenge:
    challenges: List[str] = field(default_factory=list)
    unsupported_reasoning_found: bool = False
    severity: str = "low"
    challenge_targets: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusResult:
    final_action: str  # 'present_decision', 'present_decision_with_caveats', 'acquire_more_evidence'
    headline: str
    decision_state: str
    pd_probability: float
    calibrated_confidence: float
    unresolved_concerns: List[str] = field(default_factory=list)
    decision_brief: str = ""
    traceability_matrix: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionIteration:
    iteration: int
    prediction: Dict[str, Any]
    phenotype: Dict[str, Any]
    change_report: Dict[str, Any]
    evidence_state: Dict[str, Any]
    acquisition: Optional[Dict[str, Any]] = None
    assessment_result: Optional[Dict[str, Any]] = None
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    uncertainty_reduction: float = 0.0
    cumulative_cost: float = 0.0
    cumulative_burden: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionState:
    patient_id: str
    visit_id: int
    current_prediction: PredictionResult
    predictive_uncertainty: float
    signal_quality: float
    model_agreement: float
    phenotype: PhenotypeResult
    longitudinal_state: ChangeDetectionResult
    evidence_state: EvidenceState
    accumulated_evidence: Dict[str, Any] = field(default_factory=dict)
    assessments_performed: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_burden: float = 0.0
    iteration: int = 0
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    is_terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ScenarioConfig:
    scenario_name: str = "moderate"
    class_overlap: float = 0.15
    noise_level: float = 0.10
    temporal_variability: float = 0.08
    missingness_rate: float = 0.05
    outlier_rate: float = 0.03
    contradiction_rate: float = 0.08
    longitudinal_strength: float = 0.25
    assessment_noise: float = 0.05
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
