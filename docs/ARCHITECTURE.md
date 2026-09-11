# System Architecture

## Evidence-Aware Sequential Decision Support for Parkinson's Voice Assessment

---

## Overview

The system is a **longitudinal, evidence-driven clinical decision-support framework** that transforms Parkinson's voice screening from a static one-shot classification into an iterative, multi-modal evidence evaluation loop.

Rather than emitting a hard diagnosis from a single recording, the system:
1. Evaluates the *sufficiency* of currently available evidence
2. Identifies *specific gaps* driving uncertainty
3. Selects the *optimal next assessment* to resolve those gaps at lowest cost and patient burden
4. Repeats until evidence is `SUFFICIENT` or the acquisition budget is exhausted

---

## High-Level Data Flow

```
Raw Acoustic CSV
       │
       ▼
┌──────────────────────┐
│  features.py         │   22 MDVP features → 4 phenotype axes
│  Signal Quality      │   SNR proxy, missingness rate, boundary outlier ratio
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  models.py           │   Heterogeneous calibrated ensemble
│  VoiceEnsemble       │   → P(PD), confidence_margin, model_disagreement,
│                      │     predictive_entropy H(p)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  baseline.py         │   Strict prior-visit-only baseline
│  Robust Median/MAD   │   → ChangeDetectionResult (verdict, axis_deviations,
│                      │     persistence_count)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  evidence.py         │   Deterministic rule engine
│  Evidence Engine     │   → EvidenceState: SUFFICIENT | INSUFFICIENT | CONTRADICTORY
│                      │   + dominant_gaps: [EvidenceGap, ...]
└────────┬─────────────┘
         │
  ┌──────┴───────┐
  │ SUFFICIENT?  │──YES──▶ Final Decision: STABLE / PROGRESSING / AMBIGUOUS
  └──────┬───────┘
         │ NO
         ▼
┌──────────────────────┐
│  acquisition.py      │   EIG-based utility: U(a|E) = EIG(a) − λc·C(a) − λb·B(a)
│  Next-Best Assessment│   → AcquisitionDecision (recommended assessment,
│  Policy              │     utility score, gap alignment)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  agents/             │   5 independent verifiers → AdversarialAgent → ConsensusAgent
│  Multi-Agent Layer   │   → ConsensusResult with clinician brief
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  orchestrator.py     │   Iterative loop: observe → update → decide → repeat
│  Sequential Loop     │   Tracks: entropy reduction ΔH, cost, burden, trajectory
└────────┬─────────────┘
         │
         ▼
    app.py / CLI
   (Streamlit UI / run_case.py)
```

---

## Module Descriptions

### `src/schemas.py` — Typed Data Contracts
All inter-module data is passed as typed Python dataclasses. Key types:

| Schema | Description |
|:--|:--|
| `PredictionResult` | Ensemble output: `pd_probability`, `confidence_margin`, `model_disagreement`, `predictive_entropy` |
| `SignalQualityMetrics` | `missingness_rate`, `snr_proxy`, `boundary_outlier_ratio`, `overall_quality` |
| `PhenotypeResult` | 4-axis phenotype severity: `vocal_stability`, `amplitude_variation`, `noise_characteristics`, `nonlinear_dynamics` |
| `BaselineResult` | Robust Median/MAD location and dispersion for all phenotype axes |
| `ChangeDetectionResult` | Change verdict, axis deviations, effect size, persistence count |
| `EvidenceState` | Evidentiary state + `dominant_gaps` + `uncertainty_sources` |
| `AssessmentDefinition` | Assessment metadata: cost, burden, `target_gaps` |
| `AcquisitionDecision` | Selected assessment, utility score, gap alignment rationale |
| `VerificationResult` | Per-agent structured audit outcome |
| `ConsensusResult` | Multi-agent vote + clinician brief |
| `DecisionState` | Snapshot of full decision state at each iteration |

---

### `src/features.py` — Feature Extraction & Signal Quality

- Loads raw MDVP acoustic CSV
- Groups 22 features into 4 phenotypic axes via `PHENOTYPE_COLUMNS`
- Computes `SignalQualityMetrics`:
  - `missingness_rate` — fraction of NaN values
  - `snr_proxy` — median-to-MAD ratio across features
  - `boundary_outlier_ratio` — fraction of features at clipped extremes
  - `overall_quality` — weighted composite score

---

### `src/models.py` — Calibrated Ensemble Classifier (`VoiceEnsemble`)

Three base learners:
- `LogisticRegression` (L2 regularized)
- `RandomForestClassifier` (100 trees)
- `GradientBoostingClassifier` (100 stages)

Pipeline per learner: `SimpleImputer(median)` → `StandardScaler` → `CalibratedClassifierCV(method='sigmoid')`

Outputs per prediction:
- `pd_probability` — ensemble mean probability
- `confidence_margin` — `|p − 0.5| × 2` ∈ [0,1]
- `model_disagreement` — ensemble std across individual probabilities
- `predictive_entropy` — Shannon entropy $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$
- `calibration_error` — Expected Calibration Error (ECE, 10-bin)

---

### `src/baseline.py` — Patient-Specific Change Detection

Zero future-data leakage: only visits strictly before `current_visit_id` are used.

**Change detection logic:**
1. Compute median (location) and $1.4826 \times \text{MAD}$ (dispersion) per phenotype axis
2. Compute normalized deviation: $z_{robust} = \frac{x - \hat{\mu}}{\hat{\sigma} + \varepsilon}$
3. Check $z_{robust} > 1.75$ AND effect size $> \Delta_{min} = 0.50$
4. Track temporal persistence: require deviation across ≥2 consecutive visits
5. Require multi-axis agreement: ≥2 axes showing simultaneous deviation

---

### `src/evidence.py` — Evidence State Machine

Five evidence dimensions evaluated against configurable thresholds:

| Dimension | Threshold |
|:--|:--|
| Signal quality | ≥ 0.55 |
| Confidence margin | ≥ 0.35 |
| Model disagreement | ≤ 0.12 |
| Predictive entropy | ≤ 0.88 bits |
| Phenotype consistency | ≥ 0.60 |

**State assignment rules:**
- `SUFFICIENT` — All dimensions pass
- `CONTRADICTORY` — ≥2 dimensions fail with opposing signals
- `INSUFFICIENT` — ≥1 dimension fails (default when not contradictory)

---

### `src/acquisition.py` — Assessment Space & EIG Policy

Six assessments in the feasible catalog:

| Assessment | Evidence Type | Cost | Burden |
|:--|:--|:--:|:--:|
| Repeat Sustained Vowel | acoustic_phonation | 0.05 | 0.05 |
| Standardized Reading Passage | acoustic_continuous_speech | 0.10 | 0.10 |
| DDK Task | acoustic_articulatory | 0.15 | 0.15 |
| 7-Day Home Monitoring | longitudinal_ecological | 0.30 | 0.45 |
| Accelerometer Gait Check | multimodal_inertial | 0.35 | 0.35 |
| Clinical UPDRS Exam | clinical_specialist | 0.70 | 0.80 |

**Utility function:**
$$U(a \mid E) = \text{EIG}(a) - \lambda_c \cdot C(a) - \lambda_b \cdot B(a)$$

where $\lambda_c = 0.50$, $\lambda_b = 0.30$.

---

### `src/agents/` — Multi-Agent Verification Layer

| Agent | Role |
|:--|:--|
| `SignalQualityAgent` | Audits recording quality, SNR, boundary artifacts |
| `ReliabilityAgent` | Verifies calibration, entropy, ensemble disagreement |
| `PhenotypeAgent` | Checks cross-axis phenotypic consistency |
| `TemporalAgent` | Validates temporal change detection evidence |
| `AcquisitionDecisionVerifier` | Audits gap alignment and EIG adequacy of assessment choice |
| `AdversarialAgent` | Challenges overconfident or low-quality decisions |
| `ConsensusAgent` | Aggregates 5 verifier votes into `ConsensusResult` with clinician brief |

---

### `src/agents/orchestrator.py` — Sequential Decision Loop

```python
for iteration in range(max_iterations):
    evidence_state = assess_evidence(pred, phenotype, change, signal)
    if evidence_state.state == "SUFFICIENT":
        break
    decision = select_next_assessment(evidence_state, acquired)
    observation = sample_assessment_observation(decision.recommended_assessment, evidence_state)
    evidence_state = update_evidence(evidence_state, observation)
    trajectory.append(DecisionState(...))
```

Tracks: `entropy_before`, `entropy_after`, `ΔH`, cumulative cost and burden.

---

### `src/evaluation.py` — Research Evaluation

Implements all evaluation protocols:
- ROC AUC, Brier Score, ECE (calibration)
- Selective risk-coverage curves (abstention threshold sweep)
- Change detection: Precision, Recall, F1, FPR vs ground-truth events
- Policy benchmarking: Accuracy, UDR, AvgCost, EIG/Cost efficiency
- Sequential efficiency: ΔH per iteration

---

### `src/ablation.py` — Incremental Component Ablation

9 ablation levels (A0–A8) systematically enabling components, measuring Accuracy and UDR.

---

### `src/visualizations.py` — Publication-Quality Figures

Generates all 10 research figures to `results/plots/`:

| Plot | Description |
|:--|:--|
| `01_calibration_curve.png` | Reliability diagram with ECE annotation |
| `02_risk_coverage_curve.png` | Accuracy vs coverage selective prediction |
| `03_change_detection_comparison.png` | F1/Precision/Recall bar comparison |
| `04_uncertainty_reduction_trajectory.png` | ΔH per sequential iteration |
| `05_cost_vs_uncertainty_reduction.png` | Pareto efficiency scatter |
| `06_policy_comparison.png` | 6-policy head-to-head |
| `07_decision_state_distributions.png` | Evidence state across difficulty levels |
| `08_assessment_selection_frequency.png` | Assessment usage distribution |
| `09_ablation_performance.png` | A0–A8 accuracy/UDR curves |
| `10_longitudinal_case_study.png` | Per-patient longitudinal trajectory |

---

## Entry Points

| Script | Purpose |
|:--|:--|
| `python -m experiments.run_all` | Full evaluation suite (~15 min) |
| `python run_case.py --patient P001 --visit 3 --no-llm` | Single patient sequential analysis |
| `python run_evaluation.py` | Evaluation metrics only |
| `streamlit run app.py` | Interactive dashboard |
| `pytest tests/` | Full test suite (16 tests) |

---

## Technology Stack

| Component | Library |
|:--|:--|
| ML Ensemble | scikit-learn 1.x |
| Calibration | `CalibratedClassifierCV(method='sigmoid')` |
| Data | pandas, numpy |
| Visualization | matplotlib 3.10 |
| Dashboard | Streamlit |
| Tests | pytest |
| Python | 3.13.2 |
