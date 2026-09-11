# Limitations & Known Constraints

## Evidence-Aware Sequential Decision Support for Parkinson's Voice Assessment

---

## 1. Synthetic Data Only (No Real Clinical Audio)

**What this means**: All experiments in this project use synthetically generated MDVP-style acoustic biomarker feature vectors. No actual patient voice recordings were used.

**Impact on generalizability**:
- The calibrated ensemble achieves AUC = 1.00 on the synthetic test set, which reflects the controlled generative process, not real-world clinical performance
- Real clinical audio introduces: microphone variability, ambient noise, session-to-session recording protocol drift, patient fatigue effects, medication timing confounds (levodopa ON/OFF state)
- Change detection F1 = 0.370 on synthetic data does not imply equivalent performance on real longitudinal recordings

**Mitigation path**: The data loader and `DataLoader` protocol in `src/features.py` are designed to accept real MDVP CSV exports without code changes.

---

## 2. Observation Models Are Approximate

**What this means**: The probabilistic resolution probabilities in `AssessmentDefinition.target_gaps` (e.g., "Repeat vowel resolves `low_signal_quality` with probability 0.90") are manually specified based on clinical rationale, not empirically estimated from clinical trial data.

**Impact**: The EIG computation is only as accurate as these resolution probabilities. If the actual resolution rate for a given assessment-gap pair differs significantly from the assumed value, the selected assessment may be suboptimal.

**Mitigation path**: In a real deployment, these probabilities should be estimated from historical records: "In past cases where a repeat vowel was administered after a `low_signal_quality` failure, what fraction resolved the quality issue?"

---

## 3. Change Detection Precision Is Low

**What this means**: The proposed longitudinal method achieves Precision = 0.232 (vs Recall = 0.907). This means ~77% of "change detected" alerts are false positives.

**Design choice**: This reflects a deliberate clinical safety bias toward high recall (not missing genuine progression events) at the expense of precision (many false alerts).

**Real-world impact**: In a deployed system, low precision would generate excessive clinician workload from false change alerts. Adjusting `Z_ROBUST_THRESHOLD` (currently 1.75) to 2.0 or 2.5 would improve precision at the cost of missing more genuine changes.

**Note on benchmark comparison**: The naive population threshold achieves F1 = 0.850 because the synthetic data is designed with clearly separated class distributions. On real longitudinal data with high inter-patient variability, population thresholds are expected to degrade substantially while individualized methods are expected to improve.

---

## 4. Policy Benchmarks Show Similar Performance

**What this means**: All 6 acquisition policies achieve near-identical decision accuracy and UDR = 0.000 in the benchmark. The differences are primarily in cost efficiency.

**Why this happens**: The synthetic dataset is well-behaved (most test cases reach SUFFICIENT evidence after 0–1 acquisitions). In a dataset with more genuinely ambiguous cases, the advantage of the EIG-based policy over random or fixed-order selection would be more visible.

**Real-world expectation**: The EIG policy is expected to show larger advantage when:
- Multiple competing gaps exist simultaneously
- Some assessments are unavailable (availability constraints)
- Patient burden constraints are binding

---

## 5. No Real Multi-Modal Sensor Integration

**What this means**: The `accelerometer_gait_check` and `extended_home_monitoring` assessments exist in the assessment catalog and observation models, but their actual sensor data is not processed. The system simulates what would happen if these assessments were conducted, but cannot ingest real wearable sensor streams.

**Mitigation path**: A real deployment would add a sensor data adapter module feeding into the `features.py` pipeline alongside acoustic data.

---

## 6. LLM Agent Fallback Behavior

**What this means**: When `--no-llm` is specified or no LLM is configured, the `AdversarialAgent` and `ConsensusAgent` produce deterministic structured outputs based on rule-based logic rather than language model reasoning.

**Impact**: The clinician brief in offline mode is a templated string rather than natural language narration. All clinical reasoning logic (thresholds, gap identification, consensus voting) remains deterministic and unaffected.

---

## 7. No Prospective Clinical Validation

**What this means**: This system has not been tested in a prospective clinical study. The framework is a research prototype.

**Critical constraint**: This system is explicitly designed as a **clinical decision-support tool**, not a diagnostic instrument. It should never be used to make autonomous clinical decisions without physician oversight.

---

## 8. Calibration ECE Is Non-Zero

**What this means**: The ECE = 0.0765 indicates the calibrated ensemble still has 7.65% average gap between predicted confidence and empirical accuracy.

**Impact**: Slightly overconfident probability estimates in the 0.6–0.8 range mean the evidence state engine may classify some uncertain cases as `SUFFICIENT` when they should remain `INSUFFICIENT`.

**Mitigation path**: Isotonic regression calibration (as opposed to sigmoid/Platt) may improve ECE. Temperature scaling post-hoc calibration could also be applied.

---

## 9. Minimum Baseline History Requirement

**What this means**: The robust change detection requires ≥ 2 prior visits. First-time and second-visit patients receive `verdict = "no_baseline_insufficient_history"` and the evidence state includes the `unestablished_baseline` gap.

**Impact**: For new patients, the system will more frequently recommend the 7-Day Home Monitoring assessment to bootstrap the baseline, increasing burden at enrollment.

---

## 10. Computational Cost of EIG Estimation

**What this means**: EIG is estimated via `N_MC_SAMPLES = 32` Monte Carlo draws per assessment per iteration. For a catalog of 6 assessments and 5 iterations, this is 960 samples per patient case.

**Impact**: Runtime per case is well under 1 second, but scales linearly with catalog size and iteration budget.

---

## Summary Table

| Limitation | Severity | Mitigation Status |
|:--|:--:|:--|
| Synthetic data only | High | Data loader ready for real MDVP CSV |
| Observation model probabilities are hand-specified | Medium | Requires empirical calibration from clinical records |
| Low change detection precision (0.232) | Medium | Threshold tunable; expected to improve on real data |
| No real wearable sensor integration | Medium | Sensor adapter module not yet built |
| LLM offline mode uses rule-based fallback | Low | Core logic deterministic; narration only affected |
| No prospective clinical validation | High | Research prototype; requires clinical IRB study |
| ECE = 0.0765 (non-zero calibration error) | Low | Isotonic/temperature scaling could improve |
| Minimum 2 visits for baseline | Low | Inherent requirement; mitigated by home monitoring |
