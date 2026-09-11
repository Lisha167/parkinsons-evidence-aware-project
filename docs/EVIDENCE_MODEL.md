# Evidence Model

## Deterministic Evidence State Engine

---

## Overview

The Evidence Model (`src/evidence.py`) provides a **purely deterministic, threshold-based evaluation** of whether the currently available acoustic and longitudinal evidence is sufficient to support a clinical decision.

It does not use an LLM or statistical black box. Every state classification is derived from explicit, inspectable comparisons against clinically-motivated numerical thresholds.

---

## Five Evidence Dimensions

Evidence is assessed across five independent dimensions, each corresponding to a different failure mode:

### 1. Signal Quality
- **Source**: `SignalQualityMetrics.overall_quality` from `src/features.py`
- **Threshold**: ≥ 0.55
- **Failure type**: `low_signal_quality`
- **Gap severity**: proportional to distance below threshold + 0.30 base

Recording artifacts, microphone proximity changes, or background noise degrade the SNR proxy and boundary outlier measures, yielding a signal quality score below the threshold.

### 2. Predictive Confidence Margin
- **Source**: `|pd_probability − 0.5| × 2` from `VoiceEnsemble`
- **Threshold**: ≥ 0.35
- **Failure type**: `low_predictive_confidence`
- **Gap severity**: proportional to distance below threshold + 0.20 base

When the ensemble predicts a probability close to 0.5, the confidence margin is near zero — the model is operating near its decision boundary and should not be trusted without additional evidence.

### 3. Model Disagreement (Epistemic Uncertainty)
- **Source**: `std(individual_ensemble_probs)` from `VoiceEnsemble`
- **Threshold**: ≤ 0.12
- **Failure type**: `high_model_disagreement`
- **Gap severity**: proportional to excess above threshold / 0.15 + 0.30 base

High standard deviation across ensemble members indicates genuine epistemic model uncertainty, typically driven by sparse or unusual feature combinations outside the training distribution.

### 4. Predictive Entropy (Aleatoric Uncertainty)
- **Source**: $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ from `VoiceEnsemble`
- **Threshold**: ≤ 0.88 bits (max possible = 1.0 bit)
- **Failure type**: `high_predictive_entropy`
- **Gap severity**: proportional to excess above threshold / 0.20 + 0.20 base

Shannon entropy measures intrinsic ambiguity in the probability estimate itself. High entropy (close to 1.0 bit) indicates that the current evidence is nearly maximally uninformative about the class label.

### 5. Phenotype Consistency
- **Source**: Concordance between rule-based acoustic phenotype severity and ML predicted probability
- **Threshold**: ≥ 0.60
- **Failure type**: `phenotype_mismatch`
- **Gap severity**: proportional to shortfall / 0.40 + 0.30 base

```python
concordance = 1.0 − |phenotype_severity − pd_probability|
```

When the structured acoustic phenotype analysis disagrees strongly with the ML probability, this signals either feature noise or model miscalibration.

---

## Evidence Gap Identification

The function `identify_evidence_gaps()` evaluates all five dimensions simultaneously and returns a list of `EvidenceGap` objects, sorted by severity (descending):

```python
@dataclass
class EvidenceGap:
    gap_type: str        # e.g. "low_signal_quality"
    severity: float      # 0.0 to 1.0
    description: str     # Human-readable explanation
```

Two additional temporal gaps are also checked:
- `unestablished_baseline` (severity 0.50): Fewer than 2 prior visits available
- `temporal_predictive_tension` (severity 0.70): Detected change without matching confidence

---

## State Classification Rules

Given the extracted gaps, the **evidentiary state** is assigned:

```python
n_gaps = len(gaps)
n_severe_gaps = len([g for g in gaps if g.severity > 0.60])

if n_gaps == 0:
    state = "SUFFICIENT"
elif n_gaps >= 2 and n_severe_gaps >= 2:
    # Simultaneous multiple severe failures → contradictory
    state = "CONTRADICTORY"
else:
    state = "INSUFFICIENT"
```

### `SUFFICIENT`
All five dimensions pass their thresholds. The current acoustic evidence supports a clinical decision.

### `INSUFFICIENT`
One or more dimensions fail. Evidence is inadequate but not internally contradictory. The next-best assessment should be selected to resolve the highest-severity gap.

### `CONTRADICTORY`
Two or more dimensions simultaneously fail severely (severity > 0.60). This often indicates a fundamental recording quality breakdown or a novel patient phenotype outside the training distribution. Clinical referral is the appropriate response.

---

## Uncertainty Source Attribution

The `EvidenceState` schema includes `uncertainty_sources`, which maps each active gap to its contributing dimension:

```python
uncertainty_sources = {
    "low_signal_quality": "recording_artifact_or_environment",
    "high_model_disagreement": "epistemic_model_uncertainty",
    "phenotype_mismatch": "cross_modal_acoustic_inconsistency",
    ...
}
```

This attribution is surfaced in the Streamlit dashboard and in the verification agent reports.

---

## Threshold Configuration

All thresholds are centralized in the `THRESH` dict at the top of `evidence.py`:

```python
THRESH = dict(
    min_signal_quality=0.55,
    min_confidence_margin=0.35,
    max_model_disagreement=0.12,
    max_predictive_entropy=0.88,
    min_phenotype_consistency=0.60,
)
```

These can be tightened or relaxed without changing the core classification logic.

---

## Example Evaluation

```python
from src.evidence import assess_evidence

evidence_state = assess_evidence(
    pred_summary={"pd_probability": 0.62, "confidence_margin": 0.24, ...},
    phenotype={"overall_phenotype_severity": 0.45},
    change_report={"verdict": "stable_within_normal_fluctuation"},
    signal_metrics=signal_quality_object,
)

print(evidence_state.state)        # "INSUFFICIENT"
print(evidence_state.dominant_gaps)
# [EvidenceGap("low_predictive_confidence", 0.61, "Margin 0.24 < threshold 0.35")]
```
