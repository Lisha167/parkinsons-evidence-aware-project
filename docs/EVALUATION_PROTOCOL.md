# Evaluation Protocol

## Research Evaluation Framework & Acceptance Criteria

---

## Overview

All evaluation is executed by `experiments/run_all.py`, which runs the full 8-step research evaluation pipeline deterministically with `RANDOM_SEED = 42`.

```bash
python -m experiments.run_all
```

Execution time: ~15 minutes. All outputs are saved to `results/`.

---

## Evaluation Steps

### Step 1: Multi-Difficulty Synthetic Cohort Generation

**Script**: `data/generate_synthetic_data.py`  
**Output**: 50 patients × 6 visits × 4 difficulty levels = 1,200 total records

Four difficulty scenarios are generated:

| Scenario | Class Overlap | Noise | Missingness | Outlier Rate |
|:--|:--:|:--:|:--:|:--:|
| Easy | 2% | 3% | 0% | 0% |
| Moderate | 15% | 10% | 5% | 3% |
| Hard | 30% | 18% | 12% | 8% |
| Ambiguous | 45% | 25% | 20% | 15% |

Ground-truth clinical events are injected explicitly (`known_change_event` column) with `change_magnitude`, `change_type`, and `change_start_visit` annotations.

---

### Step 2: Calibrated Ensemble Training

**Module**: `src/models.py` — `VoiceEnsemble`  
**Split**: Strict patient-level split (75% train / 25% test, no cross-visit leakage)  
**Metrics reported**:
- `ROC AUC` (discrimination)
- `Brier Score` (probabilistic accuracy)
- `ECE` (calibration: 10-bin weighted reliability)

**Result file**: `results/calibration_metrics.json`

---

### Step 3: Selective Prediction Risk-Coverage Curve

Threshold sweep over confidence abstention thresholds [0.0, 0.5]:
- For each threshold $\tau$: retain predictions with `confidence_margin ≥ τ`
- Report: (coverage fraction, accuracy on retained subset)

Expected property: monotonically increasing accuracy as coverage decreases.

**Result file**: `results/classification_metrics.json`  
**Figure**: `results/plots/02_risk_coverage_curve.png`

---

### Step 4: Evidence State Distribution

Across all 4 difficulty levels, compute the fraction of assessments classified as:
- `SUFFICIENT`
- `INSUFFICIENT`
- `CONTRADICTORY`

Expected property: Hard/Ambiguous scenarios should produce more INSUFFICIENT/CONTRADICTORY states.

**Figure**: `results/plots/07_decision_state_distributions.png`

---

### Step 5: Change Detection Benchmark

**Module**: `src/evaluation.py` — `evaluate_change_detection()`  
**Ground truth**: `known_change_event` column (injected in data generation)

Four methods compared:
1. `naive_population_threshold`: `pd_probability > 0.60`
2. `simple_patient_zscore`: Patient z-score without persistence or effect size
3. `robust_mad_baseline_only`: MAD baseline without persistence or multi-axis
4. `proposed_longitudinal_method`: Full proposed method (MAD + effect size + persistence + multi-axis)

**Metrics**: Precision, Recall, F1, FPR (False Positive Rate)

**Result file**: `results/change_detection_metrics.json`  
**Figure**: `results/plots/03_change_detection_comparison.png`

---

### Step 6: Acquisition Policy Benchmark

**Module**: `src/policies.py`  
**Policies**: 6 policies evaluated on identical test cases

For each policy:
- `decision_accuracy`: Fraction of cases where final decision matches known label
- `unsupported_decision_rate (UDR)`: Fraction of cases where policy decides with INSUFFICIENT evidence
- `avg_assessments_per_case`: Mean number of additional assessments selected
- `avg_cost`: Mean total acquisition cost
- `eig_per_cost`: Information gain per unit cost (efficiency)

**Result file**: `results/acquisition_policy_comparison.json`  
**Figure**: `results/plots/06_policy_comparison.png`

---

### Step 7: Component Ablation Study (A0–A8)

**Module**: `src/ablation.py`  
9 configurations evaluated:

| Level | Component Added |
|:--:|:--|
| A0 | Static single LogReg classifier (always decides) |
| A1 | + Heterogeneous calibrated ensemble + uncertainty |
| A2 | + Structured acoustic phenotype concordance |
| A3 | + Robust patient-specific baseline and change validation |
| A4 | + Deterministic evidence state machine |
| A5 | + EIG-based next-best assessment policy |
| A6 | + 5-agent verification layer |
| A7 | + Adversarial critic |
| A8 | Full integrated framework |

**Metrics per level**: Accuracy, UDR, Deferral/Acquisition Rate  
**Expected**: Monotonic UDR decrease from A0 to A8

**Result file**: `results/ablation_results.json`  
**Figure**: `results/plots/09_ablation_performance.png`

---

### Step 8: Sequential Orchestration & Decision Trajectories

**Module**: `src/agents/orchestrator.py`  
Runs full iterative sequential analysis for all test patients.

**Tracked per trajectory**:
- Per-iteration entropy: `H_t` → `H_{t+1}`
- Entropy reduction: `ΔH = H_t − H_{t+1}`
- Cumulative cost and burden
- Evidence state at each iteration
- Assessment selected at each iteration

**Result file**: `results/decision_trajectories.json`  
**Figures**: `04_uncertainty_reduction_trajectory.png`, `05_cost_vs_uncertainty_reduction.png`, `10_longitudinal_case_study.png`

---

## Key Metrics Definitions

| Metric | Definition |
|:--|:--|
| **ROC AUC** | Area under the ROC curve; 0.5 = random, 1.0 = perfect |
| **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$; 0.0 = perfect, 0.25 = random |
| **ECE** | $\sum_b \frac{|B_b|}{N}|acc(B_b) - conf(B_b)|$; 10 bins |
| **UDR** | Fraction of decisions made under INSUFFICIENT/CONTRADICTORY evidence |
| **EIG** | $H_{before} - \mathbb{E}[H_{after}]$ (bits) |
| **EIG/Cost** | EIG per unit acquisition cost (efficiency) |
| **ΔH** | Per-iteration entropy reduction |
| **F1 (Change)** | Harmonic mean of Precision and Recall for change detection |

---

## Reproducibility

All experiments use:
```python
RANDOM_SEED = 42
np.random.seed(42)
random.seed(42)
```

Results are deterministic across runs on the same Python environment.

---

## Output Files

After `python -m experiments.run_all`, the following are generated:

```
results/
├── FINAL_RESEARCH_REPORT.md              # Auto-generated comprehensive report
├── calibration_metrics.json              # AUC, Brier, ECE
├── classification_metrics.json           # Risk-coverage curve data
├── change_detection_metrics.json         # Per-method change detection benchmark
├── acquisition_policy_comparison.json    # 6-policy benchmark
├── ablation_results.json                 # A0–A8 ablation data
├── decision_trajectories.json            # Full sequential trajectories
├── sequential_metrics.json               # Mean ΔH, iterations per case
├── uncertainty_reduction.json            # Entropy reduction statistics
├── unsupported_decision_rate.json        # UDR per policy/configuration
└── plots/
    ├── 01_calibration_curve.png
    ├── 02_risk_coverage_curve.png
    ├── 03_change_detection_comparison.png
    ├── 04_uncertainty_reduction_trajectory.png
    ├── 05_cost_vs_uncertainty_reduction.png
    ├── 06_policy_comparison.png
    ├── 07_decision_state_distributions.png
    ├── 08_assessment_selection_frequency.png
    ├── 09_ablation_performance.png
    └── 10_longitudinal_case_study.png
```

---

## CLI Verification

After running the master evaluation, verify CLI entry points:

```bash
# Single patient analysis (no LLM required)
python run_case.py --patient P001 --visit 3 --no-llm

# Standalone evaluation metrics
python run_evaluation.py

# Full test suite
pytest tests/ -v

# Interactive dashboard
streamlit run app.py
```
