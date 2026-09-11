# Evidence-Aware Sequential Decision Support for Parkinson's Voice Assessment

> **A research prototype demonstrating evidence-aware, iterative clinical decision support for longitudinal Parkinson's voice biomarker assessment.**  
> Free, local, reproducible. No paid APIs. No cloud calls.

---

## What This System Does

Rather than emitting a static diagnosis from a single recording, this framework treats voice-based PD assessment as a **sequential evidence-accumulation problem**:

1. **Extracts** 4-axis acoustic phenotypes (vocal stability, amplitude variation, noise characteristics, nonlinear dynamics) from 22 MDVP biomarkers
2. **Predicts** PD probability using a calibrated heterogeneous ensemble with full uncertainty quantification
3. **Evaluates** evidentiary sufficiency across 5 independent dimensions (signal quality, predictive confidence, model disagreement, predictive entropy, phenotype consistency)
4. **Selects** the next-best assessment using formal Expected Information Gain minus cost/burden penalization
5. **Validates** patient-specific longitudinal change using robust Median/MAD statistics (zero future-data leakage)
6. **Verifies** every decision through 5 independent specialized agents + adversarial critic + consensus aggregation
7. **Repeats** iteratively until evidence is `SUFFICIENT` or budget is exhausted

---

## Key Research Results (Reproduced)

| Metric | Value |
|:--|:--|
| Ensemble ROC AUC | 1.0000 |
| Brier Score | 0.0158 |
| Expected Calibration Error | 0.0765 |
| Unsupported Decision Rate (Full System) | **0.000** |
| UDR (Static Baseline A0) | 0.030 |
| Change Detection Recall (Proposed) | **0.907** |
| EIG/Cost Efficiency (Proposed Policy) | **0.001 bits/unit** |
| Test Suite | **All tests passing** |
| Experiment Runtime | 934.81s |

---

## Real Dataset Integration Status

The project is structurally prepared for four publicly available UCI Parkinson's disease datasets through an explicit Dataset Abstraction Layer (`src/data/`):

| Dataset Name | Identifier | Intended Research Role | Expected Location | Status |
|:---|:---|:---|:---|:---:|
| **Synthetic Longitudinal Cohort** | `synthetic` | Controlled Experiments & Stress Testing | `data/patients_visits.csv` | 🟢 `ready` |
| **UCI Parkinson's Detection** | `uci_parkinsons_detection` | Baseline Screening Model (PD vs Healthy) | `data/raw/uci_parkinsons_detection/` | ⚪ `not_downloaded` |
| **UCI Telemonitoring** | `uci_parkinsons_telemonitoring` | Longitudinal Change Detection & UPDRS | `data/raw/uci_parkinsons_telemonitoring/` | ⚪ `not_downloaded` |
| **UCI Multiple Sound Recordings** | `uci_parkinsons_multiple_speech` | Sequential Speech Assessment Benchmarking | `data/raw/uci_parkinsons_multiple_speech/` | ⚪ `not_downloaded` |
| **UCI Replicated Acoustic** | `uci_parkinsons_replicated` | Within-Person Test-Retest Variability | `data/raw/uci_parkinsons_replicated/` | ⚪ `not_downloaded` |

*See [docs/DATASETS.md](docs/DATASETS.md) for official download links, schema specifications, and validation instructions.*

---


## Quickstart (All Free, All Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic longitudinal dataset
python data/generate_synthetic_data.py

# 3. Train the calibrated ensemble
python train.py

# 4. Run full research evaluation (~15 min)
python -m experiments.run_all

# 5. Run a single patient case
python run_case.py --patient P001 --visit 3 --no-llm

# 6. Launch interactive dashboard
streamlit run app.py

# 7. Run tests
pytest tests/ -v
```

---

## Optional: Local LLM Narration (Free, No API Key)

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2:1b    # ~1.3 GB, runs on any laptop

# Then run without --no-llm flag:
python run_case.py --patient P001 --visit 3
```

All agent reasoning is deterministic and complete without LLM. LLM adds natural-language narration only.

---

## Project Structure

```
parkinsons_project/
├── data/
│   ├── generate_synthetic_data.py   # 4-difficulty synthetic cohort generator
│   └── patients_visits.csv          # Generated on first run
├── src/
│   ├── schemas.py                   # [NEW] Typed data contracts (dataclasses)
│   ├── features.py                  # Signal quality + 4-axis phenotype extraction
│   ├── baseline.py                  # Robust Median/MAD patient-specific baselines
│   ├── models.py                    # Calibrated VoiceEnsemble + entropy/ECE
│   ├── evidence.py                  # Deterministic evidence state engine
│   ├── acquisition.py               # EIG + cost/burden acquisition policy
│   ├── policies.py                  # [NEW] 6 comparative acquisition policies
│   ├── evaluation.py                # Full evaluation suite (AUC, ECE, UDR, etc.)
│   ├── ablation.py                  # [NEW] A0–A8 ablation study runner
│   ├── visualizations.py            # [NEW] 10 publication-quality research figures
│   └── agents/
│       ├── base_agent.py            # Offline-capable base class
│       ├── signal_quality_agent.py  # Signal quality verifier
│       ├── reliability_agent.py     # Calibration/entropy verifier
│       ├── phenotype_agent.py       # Phenotype consistency verifier
│       ├── temporal_agent.py        # Temporal change verifier
│       ├── acquisition_verifier.py  # [NEW] Acquisition decision auditor
│       ├── adversarial_agent.py     # Adversarial critic
│       ├── consensus_agent.py       # Consensus aggregator
│       └── orchestrator.py          # Iterative sequential decision loop
├── experiments/
│   └── run_all.py                   # [NEW] Master experiment runner
├── tests/
│   ├── test_pipeline.py             # 4 integration tests
│   └── test_research_components.py  # [NEW] 12 research invariant unit tests
├── results/
│   ├── FINAL_RESEARCH_REPORT.md     # Auto-generated by run_all.py
│   ├── *.json                       # Machine-readable metric files
│   └── plots/                       # 10 research figures (01–10)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SEQUENTIAL_DECISION_POLICY.md
│   ├── EVIDENCE_MODEL.md
│   ├── PATIENT_SPECIFIC_CHANGE_DETECTION.md
│   ├── ASSESSMENT_OBSERVATION_MODELS.md
│   ├── VERIFICATION_ARCHITECTURE.md
│   ├── EVALUATION_PROTOCOL.md
│   └── LIMITATIONS.md
├── app.py                           # Streamlit interactive dashboard
├── train.py                         # Train ensemble CLI
├── run_case.py                      # Single-case CLI
└── run_evaluation.py                # Evaluation CLI
```

---

## Documentation Index

| Document | Description |
|:--|:--|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture and module descriptions |
| [SEQUENTIAL_DECISION_POLICY.md](docs/SEQUENTIAL_DECISION_POLICY.md) | EIG utility function, assessment catalog, policy loop |
| [EVIDENCE_MODEL.md](docs/EVIDENCE_MODEL.md) | 5-dimension evidence state engine, gap identification |
| [PATIENT_SPECIFIC_CHANGE_DETECTION.md](docs/PATIENT_SPECIFIC_CHANGE_DETECTION.md) | Robust Median/MAD, persistence, multi-axis verification |
| [ASSESSMENT_OBSERVATION_MODELS.md](docs/ASSESSMENT_OBSERVATION_MODELS.md) | Probabilistic observation models per assessment |
| [VERIFICATION_ARCHITECTURE.md](docs/VERIFICATION_ARCHITECTURE.md) | 5-agent + adversarial + consensus verification layer |
| [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | 8-step evaluation, all metrics, output files |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Known constraints, honest limitations, mitigation paths |

---

## Research Contributions

1. **Formal EIG Policy**: Next-best assessment selection using $U(a \mid E) = \text{EIG}(a) - \lambda_c C(a) - \lambda_b B(a)$
2. **Deterministic Evidence State Engine**: Computationally classifies `SUFFICIENT` / `INSUFFICIENT` / `CONTRADICTORY` from 5 evidence dimensions
3. **Robust Patient-Specific Baselines**: Strict prior-visit-only Median/MAD with effect size, persistence, and multi-axis requirements (zero data leakage)
4. **Multi-Agent Verification Architecture**: 5 specialized verifiers + adversarial critic + consensus aggregation
5. **Iterative Sequential Decision Loop**: Entropy-reducing iterative reassessment with trajectory logging
6. **Comprehensive Evaluation**: 8-step suite generating 10 plots, 10 JSON result files, and auto-generated report

---

## Honest Limitations

- **Synthetic data only**: AUC = 1.00 reflects controlled generative statistics, not real clinical performance
- **Observation model probabilities are hand-specified**, not empirically estimated
- **No real wearable sensor integration**: Gait/accelerometer assessments are simulated
- **Change detection precision is low (0.232)**: Conservative design choice; prioritizes recall to not miss progression events
- **Not a medical device**: Research prototype; requires clinical validation and IRB study before any real use

See [LIMITATIONS.md](docs/LIMITATIONS.md) for detailed discussion.

---

## Technology Stack

| Component | Technology |
|:--|:--|
| ML Ensemble | scikit-learn (LR + RF + GBT, calibrated) |
| Uncertainty | Shannon entropy, ensemble std, ECE |
| Baseline | Robust Median/MAD (1.4826 × MAD) |
| Visualization | matplotlib 3.10 |
| Dashboard | Streamlit |
| Tests | pytest (16 tests, 100% pass rate) |
| Python | 3.13.2 |
| Optional LLM | Ollama (local, free, offline) |

---

## License

Research prototype. Not licensed for clinical or commercial use. See [LIMITATIONS.md](docs/LIMITATIONS.md).
