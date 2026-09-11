# Assessment Observation Models

## Probabilistic Observation Sampling for Sequential Acquisition

---

## Overview

When the evidence state is `INSUFFICIENT` or `CONTRADICTORY`, the sequential decision policy selects the next-best assessment and simulates its likely effect on the evidence state using a **probabilistic observation model**.

The observation model encodes:
- Which evidence gaps the assessment is designed to resolve
- With what probability it actually resolves each gap (stochastic)
- What the resulting post-assessment uncertainty distribution looks like

This simulation is used to compute the **Expected Information Gain (EIG)** before deciding whether to actually recommend the assessment.

---

## Observation Model Structure

Each `AssessmentDefinition` carries a `target_gaps` mapping:

```python
@dataclass
class AssessmentDefinition:
    assessment_id: str
    name: str
    evidence_type: str
    cost: float               # Normalized [0, 1]
    burden: float             # Normalized [0, 1]
    target_gaps: Dict[str, float]   # gap_type -> resolution_probability
    description: str
```

The `target_gaps` probabilities were set based on:
1. Clinical knowledge of what each assessment actually measures
2. Conservative estimates of how often repeat assessments resolve the specific gap type

---

## Assessment Observation Profiles

### 1. Repeat Sustained Vowel (`repeat_sustained_vowel`)
**Evidence type**: `acoustic_phonation`  
**When indicated**: `low_signal_quality`, `high_predictive_entropy`

```
target_gaps = {
    "low_signal_quality":       0.90,   # Very likely to fix recording artifacts
    "high_predictive_entropy":  0.30    # Modest entropy reduction expected
}
```

*Rationale*: Most recording quality failures are transient microphone issues resolved by re-recording. Entropy reduction is partial because a re-recording of the same phonation task does not add new clinical modality information.

---

### 2. Standardized Reading Passage (`reading_passage_task`)
**Evidence type**: `acoustic_continuous_speech`  
**When indicated**: `low_predictive_confidence`, `phenotype_mismatch`, `high_predictive_entropy`

```
target_gaps = {
    "low_predictive_confidence":  0.65,
    "phenotype_mismatch":         0.70,
    "high_predictive_entropy":    0.60
}
```

*Rationale*: Connected speech activates prosodic, rate, and phonation variability features not captured by sustained vowel alone. This provides the most complementary acoustic evidence and can resolve phenotype-ML concordance failures.

---

### 3. Diadochokinetic Task (`diadochokinetic_task`)
**Evidence type**: `acoustic_articulatory`  
**When indicated**: `high_model_disagreement`, `low_predictive_confidence`, `high_predictive_entropy`

```
target_gaps = {
    "high_model_disagreement":    0.75,
    "low_predictive_confidence":  0.70,
    "high_predictive_entropy":    0.70
}
```

*Rationale*: Rapid alternating syllable repetition (/pa-ta-ka/) directly tests articulatory timing precision and motor speech rhythm, which are among the most sensitive PD-specific biomarkers. Strong gap resolution expected for model disagreement.

---

### 4. 7-Day Home Voice Diary (`extended_home_monitoring`)
**Evidence type**: `longitudinal_ecological`  
**When indicated**: `unestablished_baseline`, `temporal_predictive_tension`

```
target_gaps = {
    "unestablished_baseline":      0.85,
    "temporal_predictive_tension": 0.80
}
```

*Rationale*: The primary use case is baseline establishment. 7 days of ecological recordings provide sufficient within-person variability estimation to construct a valid individual baseline. High resolution probability (0.85) but high cost/burden.

---

### 5. Accelerometer Gait Check (`accelerometer_gait_check`)
**Evidence type**: `multimodal_inertial`  
**When indicated**: `phenotype_mismatch`, `high_model_disagreement`, `low_predictive_confidence`

```
target_gaps = {
    "phenotype_mismatch":        0.80,
    "high_model_disagreement":   0.60,
    "low_predictive_confidence": 0.65
}
```

*Rationale*: Cross-modality evidence from wearable inertial sensors (postural tremor, stride variability) can confirm or refute acoustic phenotype findings. Highly effective for resolving phenotype-ML concordance failures when voice-only evidence is ambiguous.

---

### 6. Clinical MDS-UPDRS Exam (`clinical_updrs_exam`)
**Evidence type**: `clinical_specialist`  
**When indicated**: Any high-severity gap (terminal resource)

```
target_gaps = {
    "low_signal_quality":         0.95,
    "low_predictive_confidence":  0.95,
    "high_model_disagreement":    0.95,
    "high_predictive_entropy":    0.95,
    "phenotype_mismatch":         0.95,
    "unestablished_baseline":     0.90,
    "temporal_predictive_tension": 0.90
}
```

*Rationale*: Full specialist neurological examination is the gold standard reference. The 0.95 resolution probability reflects that a thorough MDS-UPDRS-III evaluation can resolve essentially any acoustic evidence ambiguity. It is selected last due to high cost (0.70) and high burden (0.80).

---

## Sampling Procedure

The function `sample_assessment_observation()` in `src/acquisition.py` implements Monte Carlo sampling:

```python
def sample_assessment_observation(
    assessment: AssessmentDefinition,
    evidence_state: EvidenceState,
    n_samples: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> AssessmentResult:

    gap_types = [g.gap_type for g in evidence_state.dominant_gaps]

    # For each gap, sample whether this assessment resolves it
    resolved_fracs = []
    for _ in range(n_samples):
        resolved = 0
        total = len(gap_types) or 1
        for gap in gap_types:
            p_resolve = assessment.target_gaps.get(gap, 0.05)
            if rng.random() < p_resolve:
                resolved += 1
        resolved_fracs.append(resolved / total)

    expected_resolution = float(np.mean(resolved_fracs))

    # Translate to an entropy reduction
    current_entropy = evidence_state.current_entropy
    post_entropy = current_entropy * (1.0 - expected_resolution * 0.7)

    return AssessmentResult(
        assessment_id=assessment.assessment_id,
        evidence_type=assessment.evidence_type,
        resolution_rate=expected_resolution,
        expected_entropy_reduction=current_entropy - post_entropy,
        sampled_post_entropy=post_entropy,
        gaps_addressed=gap_types,
    )
```

**Key design decisions:**
- `n_samples=32` provides stable EIG estimates at low computational cost
- The 0.7 factor represents that even fully resolved gaps don't collapse entropy to zero (residual uncertainty remains)
- The baseline gap resolution probability for unregistered gaps is 0.05 (assessments have small general benefit even for non-targeted gaps)

---

## EIG Computation

From the sampled observation:

$$\text{EIG}(a \mid E) = H_{current} - \hat{H}_{post}$$

where $\hat{H}_{post}$ is the expected post-assessment entropy estimated from `n_samples` Monte Carlo draws.

This value is used directly in the utility function:
$$U(a \mid E) = \text{EIG}(a \mid E) - \lambda_c \cdot C(a) - \lambda_b \cdot B(a)$$

---

## Test Verification

The test `test_sample_assessment_observation_generates_concrete_evidence` verifies:
1. Returned `AssessmentResult` contains non-None values
2. `expected_entropy_reduction ≥ 0`
3. `resolution_rate ∈ [0, 1]`
4. `sampled_post_entropy ≤ current_entropy`

The test `test_expected_information_gain_calculation` verifies:
1. EIG = 0 when assessment has no target gap alignment
2. EIG > 0 when assessment targets an active gap
3. EIG is bounded by initial entropy
