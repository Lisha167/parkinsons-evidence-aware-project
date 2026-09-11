# Implementation Audit & Research Gap Analysis

**Project**: Evidence-Aware Sequential Decision Support for Parkinson's Voice Assessment  
**Date**: 2026-09-01  
**Status**: Completed Pre-Implementation Audit (Phase 1)

---

## 1. Executive Summary & Existing Architecture

The existing repository implements an initial prototype of a Parkinson's voice screening system. It comprises a basic acoustic feature schema, a three-model scikit-learn ensemble (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`), a preliminary patient-specific z-score baseline, a rule-based evidence assessment report, a heuristic next-assessment recommender, a multi-agent verification pipeline with local LLM fallback, and a Streamlit dashboard.

### Existing Architecture Flow
```
Raw Synthetic CSV
       ↓
[src/features.py] → Normalizes acoustic features into 4 phenotype axes
       ↓
[src/models.py]   → Fits VoiceEnsemble (Sigmoid-calibrated) with patient-level split
       ↓
[src/baseline.py] → Computes historical mean/std (prior visits only)
       ↓
[src/evidence.py] → Rule-based threshold check on margin, disagreement, quality
       ↓
[src/acquisition.py] → Heuristic target gap matching with cost penalty
       ↓
[src/agents/*]    → Multi-agent verification (Signal, Reliability, Phenotype, Temporal, Acquisition, Adversarial, Consensus)
       ↓
[app.py & run_evaluation.py] → Streamlit UI and basic evaluation script
```

---

## 2. Existing Modules & Research Functionality

| Module | Current Responsibility | Current Status & Strengths |
| :--- | :--- | :--- |
| `src/features.py` | 22 MDVP acoustic features grouped into 4 axes: `vocal_stability`, `amplitude_variation`, `noise_characteristics`, `nonlinear_dynamics`. | Clean reference range normalization; handles clipping and severity index. |
| `src/models.py` | Heterogeneous ensemble (`VoiceEnsemble`), sigmoid calibration via `CalibratedClassifierCV`, patient-level split. | Prevents patient leakage during train/test; outputs `pd_probability`, `model_disagreement`, `confidence_margin`. |
| `src/baseline.py` | Patient-specific mean/std from prior visits (`visit_id < current`). | Ensures no future-visit leakage. Uses simple z-score threshold (1.5). |
| `src/evidence.py` | Rule engine classifying `sufficient`, `insufficient`, `contradictory`. | Transparent and centralized thresholds. |
| `src/acquisition.py` | Ranks candidate assessments based on fixed heuristic target matrix. | Implements cost and burden weighting conceptually. |
| `src/agents/` | 7 agents + `orchestrator.py` with offline Ollama LLM narration and deterministic fallback. | Good architectural separation; fully local execution. |
| `data/generate_synthetic_data.py` | Generates synthetic longitudinal visits for healthy and PD cohorts. | Generates baseline acoustic features and UPDRS-like severity score. |
| `run_evaluation.py` | CLI for discrimination (AUC), calibration (ECE, Brier), selective risk curve, and change detection. | Covers essential baseline metrics. |

---

## 3. Identified Deficiencies & Research Gaps

### 3.1 Simulated Metadata vs. Real Sequential Evidence (Critique of Core Requirement)
- **Problem**: In `orchestrator.py`, extra assessments set boolean metadata flags (`_extra_temporal_confirmation = True`, `_extra_modality_confirmation = True`, `_clinical_confirmation = True`). These flags do not alter acoustic feature distributions or probabilistic beliefs.
- **Requirement**: Replace pseudo-flags with a formal `AssessmentDefinition` and explicit `ObservationModel` that samples task-specific probabilistic observations (e.g. DDK syllable rate/regularity, reading passage prosody/pitch variability, repeat vowel acoustic variance reduction, gait accelerometer tremor power, clinical motor sub-scores).
- **Update Mechanism**: Implement a Bayesian / calibrated likelihood evidence update that modifies `DecisionState`, re-estimates entropy, updates posterior probability, and tracks trajectory.

### 3.2 Heuristic vs. Formal Next-Best-Assessment Policy
- **Problem**: `src/acquisition.py` uses manually assigned heuristic coefficients (`targets: {"signal_quality": 0.9}`) without computing information-theoretic entropy or expected loss reduction.
- **Requirement**: Implement formal Expected Information Gain (EIG):
  $$EIG(a) = H(D \mid E) - \mathbb{E}_{e_a}[H(D \mid E, e_a)]$$
  and formal Expected Decision Loss Reduction Utility:
  $$U(a \mid E) = \mathbb{E}[\Delta \mathcal{L}(a \mid E)] - \lambda_{\text{cost}} \text{Cost}(a) - \lambda_{\text{burden}} \text{Burden}(a)$$
  with full modeling of candidate observation distributions.

### 3.3 Patient-Specific Baseline & Change Detection
- **Problem**: Standard sample standard deviation is sensitive to outliers; simple z-score triggers false alarms on single noisy visits; no temporal persistence criterion is enforced.
- **Requirement**:
  1. Robust baseline location: Median / trimmed mean.
  2. Robust variability: Median Absolute Deviation (MAD) with scaling factor ($1.4826 \times \text{MAD}$) and minimum variability flooring.
  3. Standardized deviation: Robust $z_{\text{MAD}} = \frac{x - \text{median}}{\text{MAD}}$.
  4. Minimum clinical effect threshold ($\Delta_{\min}$).
  5. Multi-axis phenotype agreement and temporal persistence (requiring sustained elevation across consecutive visits or EWMA/CUSUM tracking).
  6. Ground-truth benchmark: Explicitly annotate `known_change_event`, `change_magnitude`, `change_start_visit` in data generation. Compare against naive population thresholds, standard z-score, robust baseline, and temporal persistence.

### 3.4 Signal Quality & Model Uncertainty Representation
- **Problem**: Signal quality is a single synthetic scalar without decomposed missingness, outlier detection, or feature validity indicators. Predictive uncertainty is only represented as $|p - 0.5| \times 2$.
- **Requirement**: Implement unified `UncertaintyRepresentation` tracking Shannon entropy $H(p)$, mutual information / epistemic ensemble variance, confidence margin, calibrated probability, and signal quality uncertainty.

### 3.5 Verification Architecture & Acquisition Verifier
- **Problem**: No independent verifier audits the acquisition decision itself; verifiers return ad-hoc dictionary structures; adversarial critic does not audit acquisition cost efficiency or premature stopping.
- **Requirement**:
  1. Add `AcquisitionDecisionVerifier` evaluating gap alignment, expected information value, budget feasibility, and cost-benefit ratio.
  2. Define typed `VerificationResult` schemas for all verifiers (`SignalQualityVerifier`, `ReliabilityVerifier`, `PhenotypeVerifier`, `TemporalVerifier`, `AcquisitionVerifier`).
  3. Strengthen `AdversarialCritic` to challenge unsupported confidence, inappropriate acquisition, unnecessary high-cost exams, and premature stopping.
  4. `ConsensusEngine` consumes all structured verifications and adversarial challenges.

### 3.6 Evaluation Suite, Policy Benchmarks, & Ablation Studies
- **Problem**: Synthetic data is easily separable (AUC ~1.0); only one acquisition heuristic is tested; no systematic ablation (A0–A8) exists.
- **Requirement**:
  1. Configurable synthetic generator (`Easy`, `Moderate`, `Hard`, `Ambiguous`) with class overlap, noise, missingness, outliers, and contradictory signals.
  2. Implement acquisition policy benchmarks: Random, Fixed Order, Lowest Cost, Highest Info Heuristic, Gap Heuristic, Proposed Cost/Burden EIG Policy.
  3. Implement full Ablation study A0 to A8.
  4. Generate publication-quality visualization figures and comprehensive metrics artifacts (`results/*.json`, `results/plots/*.png`, `results/FINAL_RESEARCH_REPORT.md`).

---

## 4. File Modification & Addition Plan

### Files to Modify
- `data/generate_synthetic_data.py`: Add configurable scenario difficulty, ground-truth event tracking (`known_change_event`, `change_magnitude`, `change_start_visit`), assessment observation noise models.
- `src/features.py`: Add robust feature extraction interfaces, signal quality metric computations (missingness, SNR proxy, outliers, boundary validity).
- `src/models.py`: Add calibrated entropy, epistemic uncertainty, probability calibration curve generation, reliable prediction schemas.
- `src/baseline.py`: Add robust location (median), robust dispersion (MAD), minimum effect threshold, EWMA/CUSUM, temporal persistence, multi-axis change detector.
- `src/evidence.py`: Implement deterministic Evidence State Engine with typed `EvidenceState` and explicit `EvidenceGap` identification.
- `src/acquisition.py`: Implement formal `AssessmentDefinition`, observation models, Expected Information Gain (EIG), decision loss reduction, and cost/burden-aware utility.
- `src/agents/base_agent.py`: Typed schemas for agent execution and deterministic fallbacks.
- `src/agents/signal_quality_agent.py`, `reliability_agent.py`, `phenotype_agent.py`, `temporal_agent.py`, `adversarial_agent.py`, `consensus_agent.py`: Output structured `VerificationResult` dataclasses.
- `src/agents/orchestrator.py`: Implement true sequential decision trajectory loop with typed `DecisionState`, Bayesian/calibrated evidence accumulation, and history tracking.
- `src/evaluation.py`: Add policy benchmarks, change detection evaluators, ablation pipeline (A0–A8), uncertainty reduction metrics, and unsupported decision rate calculations.
- `app.py`: Enhance Streamlit UI to display sequential timelines, entropy reduction curves, evidence gap breakdown, verification cards, and policy comparisons.
- `run_evaluation.py`: Update CLI runner to execute full research suite.
- `tests/test_pipeline.py`: Expand test suite.

### Files to Add
- `src/schemas.py`: Typed Pydantic/dataclass contracts for all data structures (`DecisionState`, `AssessmentDefinition`, `AssessmentResult`, `VerificationResult`, `ConsensusResult`, `ScenarioConfig`, etc.).
- `src/agents/acquisition_verifier.py`: Dedicated verifier auditing acquisition choices.
- `src/policies.py`: Implementation of 6 acquisition policies (Random, Fixed, Lowest-Cost, Greedy-Info, Gap-Heuristic, Proposed Cost-Burden EIG).
- `src/ablation.py`: Formal implementation of A0–A8 ablation configurations and benchmark evaluator.
- `src/visualizations.py`: Automated generator of the 10 required research figures.
- `experiments/run_all.py`: Single reproducible CLI command running all evaluations, generating metrics JSONs, plots, and final markdown report.
- `tests/test_research_components.py`: Unit and integration tests for entropy calculation, baseline persistence, EIG, acquisition verifier, policy benchmark, data leakage, and reproducibility.
- `docs/ARCHITECTURE.md`, `docs/SEQUENTIAL_DECISION_POLICY.md`, `docs/EVIDENCE_MODEL.md`, `docs/PATIENT_SPECIFIC_CHANGE_DETECTION.md`, `docs/ASSESSMENT_OBSERVATION_MODELS.md`, `docs/VERIFICATION_ARCHITECTURE.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/LIMITATIONS.md`.
- `results/FINAL_RESEARCH_REPORT.md`.

---

## 5. Migration Strategy

1. **Maintain Backwards Compatibility**: Ensure existing CLI arguments, import paths, and core function signatures remain compatible.
2. **Deterministic Reproducibility**: All experiments, observation sampling, and ensemble initializations take explicit random seeds.
3. **Strict Invariants**:
   - Zero future-visit leakage in baseline calculation.
   - Zero acquisition claimed without generating a concrete `AssessmentResult`.
   - Formal computation of Information Gain and Entropy reduction.
   - Deterministic programmatic decisions: LLMs provide natural-language explanation only, never ungrounded decision overrides.
