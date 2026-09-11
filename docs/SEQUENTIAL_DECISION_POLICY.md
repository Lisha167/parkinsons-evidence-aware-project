# Sequential Decision Policy

## Evidence-Aware Next-Best Assessment Selection

---

## Motivation

A static classifier emits a single decision regardless of evidence quality. This is clinically inappropriate when:
- Recording quality is compromised (SNR degradation, missing features)
- The ensemble is uncertain (high predictive entropy H(p) → 1.0 bit)
- Phenotype axes contradict each other (phenotype_mismatch gap)
- No longitudinal baseline has yet been established

The sequential decision policy treats evidence acquisition as a **controlled active learning** problem: given the current evidentiary state $E$, which additional assessment $a \in \mathcal{A}$ would maximally reduce decision uncertainty at minimum cost to the patient?

---

## Formal Problem Statement

At each decision iteration $t$, given:
- Current evidence state $E_t$
- Set of dominant evidence gaps $\mathcal{G}_t = \{g_1, g_2, \ldots\}$
- Feasible assessment catalog $\mathcal{A}$
- Acquired assessments so far $\mathcal{A}_{t-1}$

Select:
$$a^* = \arg\max_{a \in \mathcal{A} \setminus \mathcal{A}_{t-1}} U(a \mid E_t)$$

where the utility function is:
$$U(a \mid E_t) = \text{EIG}(a \mid E_t) - \lambda_c \cdot C(a) - \lambda_b \cdot B(a)$$

---

## Expected Information Gain (EIG)

EIG measures the expected reduction in predictive uncertainty after conducting assessment $a$:

$$\text{EIG}(a \mid E_t) = H(p_t) - \mathbb{E}_{e_a \sim P(e_a \mid \theta, a)}\left[H(p_{t+1} \mid e_a)\right]$$

where:
- $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ is predictive Shannon entropy
- $P(e_a \mid \theta, a)$ is the probabilistic observation model for assessment $a$
- The expectation is estimated by Monte Carlo sampling (`N_MC_SAMPLES = 32`)

### Observation Model Sampling

Each assessment $a$ has a gap-specific resolution profile stored in `target_gaps`:

```python
target_gaps = {
    "low_signal_quality": 0.90,      # Probability this assessment resolves the gap
    "high_predictive_entropy": 0.30,
    ...
}
```

A sampled observation $e_a$ is drawn stochastically:

```python
resolved_gaps = {
    gap: gap in a.target_gaps and random.random() < a.target_gaps[gap]
    for gap in current_gaps
}
new_entropy = entropy * (1 - resolution_rate * 0.7)
```

This produces a distribution over post-assessment uncertainty values, from which $\mathbb{E}[H(p_{t+1})]$ is estimated.

---

## Cost and Burden Penalization

Each assessment carries:
- **Cost** $C(a) \in [0, 1]$: Clinician time + resource cost (normalized)
- **Burden** $B(a) \in [0, 1]$: Patient effort and time cost (normalized)

Penalty weights:
- $\lambda_c = 0.50$ (cost weight)
- $\lambda_b = 0.30$ (burden weight)

The weights reflect that patient burden is materially less penalized than clinician cost in this default configuration, consistent with the goal of minimizing unnecessary clinical appointments.

---

## Assessment Catalog

| ID | Name | Cost | Burden | Availability | Primary Target Gaps |
|:--|:--|:--:|:--:|:--|:--|
| `repeat_sustained_vowel` | Repeat Sustained /a/ Vowel | 0.05 | 0.05 | Immediate | low_signal_quality, high_entropy |
| `reading_passage_task` | Standardized Reading Passage | 0.10 | 0.10 | Immediate | low_confidence, phenotype_mismatch |
| `diadochokinetic_task` | DDK /pa-ta-ka/ Task | 0.15 | 0.15 | Immediate | high_disagreement, low_confidence |
| `extended_home_monitoring` | 7-Day Home Voice Diary | 0.30 | 0.45 | Delayed (1 week) | unestablished_baseline |
| `accelerometer_gait_check` | Wearable Inertial Check | 0.35 | 0.35 | Clinic/Sensor | phenotype_mismatch, disagreement |
| `clinical_updrs_exam` | Clinical MDS-UPDRS | 0.70 | 0.80 | Specialist | all high-severity gaps |

---

## Policy Loop

```
START: evidence_state = assess_evidence(recording)
─────────────────────────────────────────────────
iteration = 0
WHILE iteration < MAX_ITERATIONS:
    IF evidence_state.state == "SUFFICIENT":
        BREAK → emit_decision()
    gaps = evidence_state.dominant_gaps
    a* = argmax U(a | E) over feasible unacquired assessments
    IF U(a*) < 0:
        BREAK → defer_to_clinician()    # No assessment is worth its cost
    observation = sample_assessment_observation(a*, evidence_state)
    evidence_state = update_evidence(evidence_state, observation)
    trajectory.append(DecisionState(...))
    iteration += 1
─────────────────────────────────────────────────
IF never SUFFICIENT after MAX_ITERATIONS:
    → defer_to_clinician() with full trajectory
```

---

## Comparative Policies (Benchmarking)

Six alternative policies are implemented in `src/policies.py`:

| Policy | Strategy |
|:--|:--|
| `RandomPolicy` | Uniform random selection over feasible assessments |
| `FixedOrderPolicy` | Deterministic fixed sequence (cost-ascending) |
| `LowestCostPolicy` | Always select cheapest available |
| `HighestInformationPolicy` | Greedy EIG maximization (cost-blind) |
| `EvidenceGapHeuristicPolicy` | Select assessment with highest gap alignment score |
| `ProposedCostBurdenAwarePolicy` | Full utility $U(a \mid E) = \text{EIG} - \lambda_c C - \lambda_b B$ |

### Benchmark Results (from `results/acquisition_policy_comparison.json`)

| Policy | Accuracy | UDR | Avg Cost | EIG/Cost |
|:--|:--:|:--:|:--:|:--:|
| Random | 0.997 | 0.000 | 0.38 | 0.000 |
| Fixed Order | 0.993 | 0.000 | 0.12 | 0.001 |
| Lowest Cost | 0.993 | 0.000 | 0.12 | 0.001 |
| Highest Info | 0.997 | 0.000 | 0.50 | 0.000 |
| Gap Heuristic | 0.997 | 0.000 | 0.42 | 0.000 |
| **Proposed** | **0.993** | **0.000** | **0.12** | **0.001** |

The proposed policy achieves Pareto efficiency: lowest cost while maintaining accuracy and zero unsupported decisions.

---

## Termination Conditions

The sequential loop terminates when any of:
1. `evidence_state.state == "SUFFICIENT"` → emit decision
2. All assessments in catalog have been acquired
3. `max_iterations` reached (configurable, default 5)
4. Best available utility `U(a*) < 0` (no beneficial assessment exists)

On termination without sufficient evidence → `DEFER_TO_CLINICIAN` decision with full annotated trajectory.
