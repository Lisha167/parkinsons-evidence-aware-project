# Real Dataset Integration Guide & Schema Specifications

## Evidence-Aware Sequential Decision Support for Parkinson's Voice Assessment

---

## 1. Overview & Research Principles

This document specifies the integration architecture for four publicly available UCI Parkinson's disease datasets alongside the existing synthetic cohort benchmark.

### Critical Research Rules Enforced:
1. **Zero Data Fabrication**: Unavailable variables in a dataset remain `None`/unobserved. Missing values are never filled with synthetic defaults or artificial placeholders.
2. **No Incompatible Merging**: The datasets are evaluated according to their distinct clinical designs; they are never combined into a single blended CSV merely to inflate sample size.
3. **Strict Subject-Level Leakage Prevention**: When a subject has multiple recordings, all recordings belonging to that subject are kept strictly within train or test.
4. **Strict Temporal Isolation**: Longitudinal baselines only use prior observations ($t < t_{\text{current}}$). Future observations never contaminate baseline estimation.
5. **Clear Operational Status**: Real datasets are marked as `not_downloaded` until actual raw files are placed into their respective directories.

---

## 2. Supported Datasets

| Dataset Identifier | Dataset Name | Official Source | Research Role | Has Longitudinal? | Has UPDRS? | Current Status |
|:---|:---|:---|:---|:---:|:---:|:---:|
| `synthetic` | Synthetic Longitudinal Cohort | `data/generate_synthetic_data.py` | Controlled Experiments & Stress Testing | Yes | Yes | 🟢 `ready` |
| `uci_parkinsons_detection` | UCI Parkinson's Disease Detection | [UCI #174](https://archive.ics.uci.edu/dataset/174/parkinsons) | Baseline Acoustic Screening (PD vs Healthy) | No | No | ⚪ `not_downloaded` |
| `uci_parkinsons_telemonitoring` | UCI Parkinson's Telemonitoring | [UCI #189](https://archive.ics.uci.edu/dataset/189/parkinson) | Longitudinal Change Detection & UPDRS Progression | Yes | Yes | ⚪ `not_downloaded` |
| `uci_parkinsons_multiple_speech` | UCI Multiple Sound Recordings | [UCI #301](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings) | Sequential Speech Assessment & Task Comparison | No | Yes | ⚪ `not_downloaded` |
| `uci_parkinsons_replicated` | UCI Replicated Acoustic Features | [UCI #489](https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features) | Within-Person Test-Retest Variability | No | No | ⚪ `not_downloaded` |

---

## 3. Dataset Specifications

### Dataset 1 — UCI Parkinson's Disease Detection
- **Citation**: Little, M., McSharry, P., Roberts, S., Costello, D., & Moroz, I. (2007).
- **URL**: https://archive.ics.uci.edu/dataset/174/parkinsons
- **Expected File**: `data/raw/uci_parkinsons_detection/parkinsons.data`
- **Clinical Cohort**: 31 individuals (23 with PD, 8 healthy controls), 195 sustained vowel recordings.
- **Role in Research**: Trains and validates baseline classifier discrimination ($P(\text{PD})$) and ECE calibration curves without longitudinal history.
- **Available Features**: 22 MDVP features: Fo, Fhi, Flo, Jitter (%, Abs, RAP, PPQ, DDP), Shimmer (dB, APQ3, APQ5, APQ11, DDA), NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE.
- **Unavailable**: UPDRS motor score, longitudinal visit timestamps, SNR proxy.

### Dataset 2 — UCI Parkinson's Telemonitoring
- **Citation**: Tsanas, A., Little, M., McSharry, P., & Ramig, L. (2009).
- **URL**: https://archive.ics.uci.edu/dataset/189/parkinson
- **Expected File**: `data/raw/uci_parkinsons_telemonitoring/parkinsons_updrs.data`
- **Clinical Cohort**: 42 early-stage PD patients tracked longitudinally via at-home telemonitoring over 6 months (~5,875 recordings).
- **Role in Research**: Evaluates patient-specific baseline drift, Median/MAD robust change detection, temporal persistence across visits, and correlation with motor UPDRS.
- **Available Features**: 16 MDVP features, `subject#`, `test_time` (days from trial onset), `motor_UPDRS`, `total_UPDRS`.
- **Unavailable**: Pitch bounds (`mdvp_fo`, `mdvp_fhi`, `mdvp_flo`), nonlinear dynamics (`spread1`, `spread2`, `d2`).

### Dataset 3 — UCI Multiple Types of Sound Recordings
- **Citation**: Sakar, B., Isenkul, M., Sakar, C., Sertbas, A., Gurgen, F., Delil, S., Apaydin, H., Kursun, O. (2013).
- **URL**: https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings
- **Expected Files**: `data/raw/uci_parkinsons_multiple_speech/train_data.txt`, `test_data.txt`
- **Clinical Cohort**: 40 subjects (20 PD, 20 controls), 26 distinct speech tasks per subject (sustained vowels, words, sentences).
- **Role in Research**: Evaluates sequential information acquisition policy across distinct acoustic tasks (sustained vowel vs. reading words vs. continuous sentences).
- **Available Features**: 26 acoustic features per task, subject IDs, UPDRS score.
- **Unavailable**: Longitudinal visit intervals (recordings performed in single multi-task session).

### Dataset 4 — UCI Replicated Acoustic Features
- **Citation**: Naranjo, L., Perez, C., Martin, J., & Campos-Roca, Y. (2016).
- **URL**: https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features
- **Expected File**: `data/raw/uci_parkinsons_replicated/ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
- **Clinical Cohort**: 80 subjects (40 PD, 40 healthy), 3 repeated recordings per subject.
- **Role in Research**: Quantifies within-subject test-retest variance ($\sigma^2_{\text{within}}$) to calibrate the variance flooring ($\varepsilon$) in robust baseline estimation.
- **Available Features**: Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE with explicit replication numbers (1, 2, 3).
- **Unavailable**: Longitudinal progression, UPDRS score.

---

## 4. Canonical Representation (`CanonicalRecord`)

The central research engine processes data through the unified typed record:

```python
@dataclass
class CanonicalRecord:
    dataset_id: str
    subject_id: str
    recording_id: Optional[str] = None
    session_id: Optional[str] = None
    visit_id: Optional[int] = None
    timestamp: Optional[str] = None
    diagnosis: Optional[int] = None          # 0=healthy, 1=PD, None=unknown
    task_type: Optional[str] = None          # "sustained_vowel", "reading", etc.
    recording_type: Optional[str] = None
    acoustic_features: Dict[str, Optional[float]]
    clinical_measurements: Dict[str, Optional[float]]
    motor_updrs: Optional[float] = None
    total_updrs: Optional[float] = None
    signal_quality: Optional[float] = None
    demographic_metadata: Dict[str, Any]
    source_metadata: Dict[str, Any]
```

---

## 5. Dataset Management CLI

The CLI provides commands to inspect, validate, and check status:

```bash
# List all registered datasets
python -m src.data.cli list

# Check readiness status
python -m src.data.cli status

# Validate raw source files
python -m src.data.cli validate uci_parkinsons_detection

# Inspect dataset manifest, supported assessments, and feature availability
python -m src.data.cli info uci_parkinsons_telemonitoring
```

---

## 6. How to Integrate Downloaded Real Datasets

When actual raw files are downloaded:
1. Place `parkinsons.data` into `data/raw/uci_parkinsons_detection/`
2. Place `parkinsons_updrs.data` into `data/raw/uci_parkinsons_telemonitoring/`
3. Place `train_data.txt` and `test_data.txt` into `data/raw/uci_parkinsons_multiple_speech/`
4. Place `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` into `data/raw/uci_parkinsons_replicated/`
5. Run `python -m src.data.cli validate all` to verify schema conformance.
6. The dataset status will transition from `not_downloaded` to `validated` automatically without code changes.
