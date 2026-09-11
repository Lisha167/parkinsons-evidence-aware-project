# Verification Architecture

## Multi-Agent Evidence Verification System

---

## Overview

All clinical evidence assessments pass through a **5-agent independent verification layer** before a final consensus decision is produced. This is followed by an adversarial critic and a consensus aggregator.

No single verification agent has veto power. The final decision is governed by the `ConsensusAgent` which applies majority voting and tracks dissent.

---

## Architecture Diagram

```
Evidence State + Prediction Summary + Acquisition Decision
        │
        ├──▶ SignalQualityAgent     → VerificationResult
        ├──▶ ReliabilityAgent       → VerificationResult
        ├──▶ PhenotypeAgent         → VerificationResult
        ├──▶ TemporalAgent          → VerificationResult
        └──▶ AcquisitionDecisionVerifier → VerificationResult
                        │
                        ▼
               AdversarialAgent (reviews all 5) → AdversarialChallenge
                        │
                        ▼
               ConsensusAgent (aggregates) → ConsensusResult
                        │
                        ▼
               Clinician Brief + Decision
```

---

## Agent Specifications

### 1. `SignalQualityAgent`
**File**: `src/agents/signal_quality_agent.py`

**Responsibility**: Audits whether the acoustic recording quality is sufficient for clinical interpretation.

**Checks performed**:
- `overall_quality < 0.55` → FAIL: recording quality below minimum threshold
- `missingness_rate > 0.20` → FAIL: excessive missing acoustic features
- `snr_proxy < 2.0` → WARN: low signal-to-noise conditions
- `boundary_outlier_ratio > 0.25` → WARN: unusual acoustic boundary patterns

**Output**: `VerificationResult(agent="signal_quality", passed=bool, confidence=float, issues=List[str], recommendation=str)`

---

### 2. `ReliabilityAgent`
**File**: `src/agents/reliability_agent.py`

**Responsibility**: Verifies that the ML ensemble produces reliable, well-calibrated predictions.

**Checks performed**:
- `predictive_entropy > 0.90` → FAIL: prediction is near-maximally uncertain
- `model_disagreement > 0.15` → FAIL: ensemble members disagree excessively
- `confidence_margin < 0.20` → FAIL: decision boundary margin too narrow for clinical use
- `calibration_error > 0.15` → WARN: ensemble calibration is poor

**Output**: `VerificationResult(agent="reliability", ...)`

---

### 3. `PhenotypeAgent`
**File**: `src/agents/phenotype_agent.py`

**Responsibility**: Checks cross-axis acoustic phenotype consistency and concordance with ML output.

**Checks performed**:
- Phenotype-ML concordance < 0.50 → FAIL: structured acoustic analysis contradicts ML prediction
- All 4 phenotype axes have low severity (<0.30) but ML predicts PD (>0.70) → FAIL: phenotype-prediction tension
- Phenotype severity is high (>0.70) but ML predicts no PD (<0.30) → FAIL: reversed phenotype-prediction tension

**Output**: `VerificationResult(agent="phenotype", ...)`

---

### 4. `TemporalAgent`
**File**: `src/agents/temporal_agent.py`

**Responsibility**: Validates whether the temporal change detection is consistent with the current prediction.

**Checks performed**:
- `verdict == "no_baseline_insufficient_history"` → WARN: no personal history to contextualize
- `verdict == "meaningful_patient_specific_change"` AND `confidence_margin < 0.40` → FAIL: detected change unsupported by confident prediction
- `verdict == "stable_within_normal_fluctuation"` AND `pd_probability > 0.80` → NOTE: stable acoustic but high PD probability (possible slow progression)

**Output**: `VerificationResult(agent="temporal", ...)`

---

### 5. `AcquisitionDecisionVerifier`
**File**: `src/agents/acquisition_verifier.py`

**Responsibility**: Independently audits whether the proposed next assessment is well-justified given the current evidence gaps.

**Checks performed**:
- Gap alignment: Does the recommended assessment target ≥1 active evidence gap?
- EIG adequacy: Is `expected_entropy_reduction > 0.05` bits (non-trivial information gain)?
- Cost/burden proportionality: Is `cost + burden < 2 × severity_of_top_gap`?
- Repeated acquisition: Is this assessment already in the acquired set?

**Output**: `VerificationResult(agent="acquisition_verifier", ...)`

---

## Adversarial Agent

**File**: `src/agents/adversarial_agent.py`

The adversarial agent reviews all 5 verification results and actively challenges potentially overconfident or low-quality decisions.

**Challenge conditions**:
- ≥ 2 verification agents failed → Challenge: "Multiple independent verifiers failed"
- `confidence_margin > 0.90` but signal quality is below threshold → Challenge: "Overconfidence with poor signal"
- All 5 agents passed but `predictive_entropy > 0.70` → Challenge: "Verification passed despite elevated entropy"

**Output**: `AdversarialChallenge(issued=bool, severity=str, challenge_text=str, specific_concerns=List[str])`

---

## Consensus Agent

**File**: `src/agents/consensus_agent.py`

Aggregates the 5 verification results and the adversarial challenge into a final `ConsensusResult`.

**Voting logic**:
```python
n_passed = sum(v.passed for v in verifier_results)
n_failed = len(verifier_results) - n_passed

if n_failed == 0 and not adversarial.issued:
    consensus = "PROCEED"       # All clear
elif n_failed >= 3:
    consensus = "REJECT"        # Majority fail → defer to clinician
elif adversarial.issued and adversarial.severity == "HIGH":
    consensus = "CAUTION"       # Adversarial challenge overrides
else:
    consensus = "CAUTION"       # Mixed results → proceed with caveats
```

**Output**:
```python
@dataclass
class ConsensusResult:
    consensus: str                        # "PROCEED" | "CAUTION" | "REJECT"
    n_agents_passed: int
    n_agents_failed: int
    adversarial_challenge_issued: bool
    clinician_brief: str                  # Human-readable summary
    supporting_evidence: List[str]
    dissenting_evidence: List[str]
    recommended_action: str
```

---

## Offline Fallback Mode

All agents inherit from `BaseAgent` (`src/agents/base_agent.py`), which detects LLM availability.

When `--no-llm` is specified or no LLM is configured, all agents operate in **deterministic offline mode**: structured verification results are computed purely from the typed evidence state without any external API call.

This ensures the verification layer functions in clinical environments without internet access.

---

## Integration Test

The test `test_acquisition_decision_verifier` verifies:
1. Verifier correctly passes a well-justified assessment choice (high gap alignment, positive EIG)
2. Verifier correctly fails an unjustified choice (no gap alignment, negligible EIG)
3. Output is a valid `VerificationResult` dataclass

The test `test_adversarial_critic_triggers_on_overconfidence` verifies:
1. Adversarial agent issues a challenge when `confidence_margin` is high but `signal_quality` is low
2. Challenge severity is `"HIGH"` in this scenario
3. `challenge_text` is a non-empty string
