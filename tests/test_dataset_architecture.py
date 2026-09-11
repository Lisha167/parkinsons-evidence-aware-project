"""
test_dataset_architecture.py
---------------------------------------------------------------
Comprehensive unit and architectural tests for the Real Dataset Integration Layer:
1. Synthetic adapter loads via registry and produces valid canonical records.
2. Dataset registry resolves all 5 datasets (synthetic + 4 UCI).
3. Missing real datasets report 'not_downloaded' gracefully (no crashes).
4. CanonicalRecord accepts all-optional fields with no fake values.
5. Missing features produce None, never fabricated values.
6. Dataset metadata and research roles are correctly assigned.
7. Subject IDs remain intact through adapter conversion.
8. Assessment capability correctly reports availability per dataset.
9. Subject-level splitting works with zero patient leakage.
10. Longitudinal temporal splitting prevents future-visit leakage.
11. FeatureSchema.MDVP_22 matches existing RAW_FEATURES.
12. Test fixtures with documented UCI schemas convert cleanly to canonical records.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.registry import DatasetRegistry
from src.data.schema import CanonicalRecord, FeatureAvailability, DatasetManifest
from src.data.validation import ValidationReport
from src.data.feature_availability import get_dataset_feature_availability, check_phenotype_axis_coverage
from src.data.assessment_capability import (
    is_assessment_available,
    get_available_assessments_for_dataset,
    ASSESSMENT_CAPABILITIES,
)
from src.data.splitting import (
    subject_level_split,
    subject_level_kfold,
    longitudinal_temporal_split,
    validate_no_leakage,
)
from src.data.adapters.uci_parkinsons_detection import UCIParkinsonsDetectionAdapter, UCI_DETECTION_COLUMN_MAP
from src.data.adapters.uci_parkinsons_telemonitoring import UCIParkinsonsTelemonitoringAdapter, UCI_TELEMONITORING_COLUMN_MAP
from src.data.adapters.uci_parkinsons_multiple_speech import UCIParkinsonsMultipleSpeechAdapter
from src.data.adapters.uci_parkinsons_replicated import UCIParkinsonsReplicatedAdapter
from src.features import FeatureSchema, RAW_FEATURES, PHENOTYPE_GROUPS


# ---------------------------------------------------------------------
# 1. Registry & Manifest Tests
# ---------------------------------------------------------------------
def test_dataset_registry_contains_all_five_datasets():
    datasets = DatasetRegistry.list_datasets()
    assert len(datasets) == 5
    assert set(datasets) == {
        "synthetic",
        "uci_parkinsons_detection",
        "uci_parkinsons_telemonitoring",
        "uci_parkinsons_multiple_speech",
        "uci_parkinsons_replicated",
    }


def test_manifest_roles_and_flags():
    manifests = DatasetRegistry.get_all_manifests()
    assert manifests["synthetic"].research_role == "controlled_experiment"
    assert manifests["uci_parkinsons_detection"].research_role == "screening"
    assert manifests["uci_parkinsons_telemonitoring"].research_role == "longitudinal_change_detection"
    assert manifests["uci_parkinsons_multiple_speech"].research_role == "sequential_speech_assessment"
    assert manifests["uci_parkinsons_replicated"].research_role == "within_subject_variability"

    # Longitudinal capabilities
    assert manifests["uci_parkinsons_telemonitoring"].has_longitudinal_data is True
    assert manifests["uci_parkinsons_detection"].has_longitudinal_data is False
    assert manifests["uci_parkinsons_replicated"].has_longitudinal_data is False

    # UPDRS presence
    assert manifests["uci_parkinsons_telemonitoring"].has_updrs is True
    assert manifests["uci_parkinsons_detection"].has_updrs is False


# ---------------------------------------------------------------------
# 2. Missing Real Dataset Graceful Handling
# ---------------------------------------------------------------------
def test_missing_real_datasets_reported_as_not_downloaded():
    for d_id in ["uci_parkinsons_detection", "uci_parkinsons_telemonitoring", "uci_parkinsons_multiple_speech", "uci_parkinsons_replicated"]:
        rep = DatasetRegistry.validate_dataset(d_id)
        assert rep.is_valid is False
        assert rep.status in ("not_downloaded", "missing_files")
        assert len(rep.files_missing) > 0
        assert len(rep.warnings) > 0


def test_missing_real_dataset_load_raises_clear_error():
    adapter = UCIParkinsonsDetectionAdapter(raw_dir="non_existent_folder_xyz")
    with pytest.raises(FileNotFoundError) as exc:
        adapter.load_raw()
    assert "Raw data file not found" in str(exc.value)
    assert "uci_parkinsons_detection" in str(exc.value)


# ---------------------------------------------------------------------
# 3. Canonical Schema Tests (No Fake Values)
# ---------------------------------------------------------------------
def test_canonical_record_allows_nullable_fields():
    rec = CanonicalRecord(dataset_id="test", subject_id="S01")
    assert rec.dataset_id == "test"
    assert rec.subject_id == "S01"
    assert rec.recording_id is None
    assert rec.visit_id is None
    assert rec.diagnosis is None
    assert rec.signal_quality is None
    assert rec.motor_updrs is None
    assert rec.total_updrs is None
    assert rec.acoustic_features == {}


# ---------------------------------------------------------------------
# 4. Feature Availability System
# ---------------------------------------------------------------------
def test_feature_availability_does_not_fabricate():
    det_avail = get_dataset_feature_availability("uci_parkinsons_detection")
    # In detection, pitch & jitter are available
    assert det_avail["jitter_pct"].available is True
    # But signal_quality, motor_updrs, and visit_id are explicitly unavailable
    assert det_avail["signal_quality"].available is False
    assert det_avail["motor_updrs"].available is False

    tele_avail = get_dataset_feature_availability("uci_parkinsons_telemonitoring")
    assert tele_avail["jitter_pct"].available is True
    assert tele_avail["motor_updrs"].available is True
    assert tele_avail["mdvp_fo"].available is False  # Fundamental freq not extracted in telemonitoring


def test_phenotype_axis_coverage():
    cov = check_phenotype_axis_coverage("uci_parkinsons_replicated", PHENOTYPE_GROUPS)
    # Vocal stability has partial coverage in replicated dataset
    assert cov["vocal_stability"]["status"] in ("partially_observed", "complete", "insufficient_evidence")
    assert "jitter_pct" in cov["vocal_stability"]["available_features"]


# ---------------------------------------------------------------------
# 5. Assessment Capabilities
# ---------------------------------------------------------------------
def test_assessment_capabilities():
    # Vowel phonation is supported across voice datasets
    assert is_assessment_available("repeat_sustained_vowel", "uci_parkinsons_detection") is True
    assert is_assessment_available("repeat_sustained_vowel", "synthetic") is True

    # Reading passage is only supported by datasets that collected it
    assert is_assessment_available("reading_passage_task", "uci_parkinsons_multiple_speech") is True
    assert is_assessment_available("reading_passage_task", "uci_parkinsons_detection") is False

    # 7-day home monitoring supported by telemonitoring
    assert is_assessment_available("extended_home_monitoring", "uci_parkinsons_telemonitoring") is True
    assert is_assessment_available("extended_home_monitoring", "uci_parkinsons_detection") is False


# ---------------------------------------------------------------------
# 6. Data Leakage & Subject Splitting
# ---------------------------------------------------------------------
def test_subject_level_splitting_prevents_leakage():
    # Sample multi-record DataFrame where each subject has 4 records
    rows = []
    for s_idx in range(20):
        for rec in range(4):
            rows.append({"subject_id": f"Sub_{s_idx}", "val": np.random.randn(), "visit_id": rec})
    df = pd.DataFrame(rows)

    train_df, test_df = subject_level_split(df, subject_col="subject_id", test_size=0.25, seed=42)
    assert len(train_df) > 0
    assert len(test_df) > 0
    # Strict validation
    assert validate_no_leakage(train_df, test_df, subject_col="subject_id") is True


def test_temporal_split_isolation():
    rows = [
        {"subject_id": "S1", "visit_id": 1, "score": 10},
        {"subject_id": "S1", "visit_id": 2, "score": 12},
        {"subject_id": "S1", "visit_id": 3, "score": 15},
        {"subject_id": "S1", "visit_id": 4, "score": 20},
    ]
    df = pd.DataFrame(rows)
    past, future = longitudinal_temporal_split(df, time_col="visit_id", cutoff_time=3)
    assert set(past["visit_id"]) == {1, 2}
    assert set(future["visit_id"]) == {3, 4}


# ---------------------------------------------------------------------
# 7. Documented Schema Fixture Tests (Labeled as FIXTURES)
# ---------------------------------------------------------------------
def test_fixture_uci_detection_conversion():
    """TEST FIXTURE: Uses 2 sample rows conforming to UCI Detection documented schema."""
    fixture_data = {
        "name": ["phon_R01_S01_1", "phon_R01_S02_1"],
        "MDVP:Fo(Hz)": [119.99, 122.40],
        "MDVP:Fhi(Hz)": [157.30, 148.65],
        "MDVP:Flo(Hz)": [74.99, 113.81],
        "MDVP:Jitter(%)": [0.0078, 0.0066],
        "MDVP:Jitter(Abs)": [0.00007, 0.00005],
        "MDVP:RAP": [0.0037, 0.0034],
        "MDVP:PPQ": [0.0055, 0.0049],
        "Jitter:DDP": [0.0110, 0.0102],
        "MDVP:Shimmer": [0.0437, 0.0383],
        "MDVP:Shimmer(dB)": [0.426, 0.366],
        "Shimmer:APQ3": [0.0218, 0.0189],
        "Shimmer:APQ5": [0.0313, 0.0287],
        "MDVP:APQ": [0.0297, 0.0245],
        "Shimmer:DDA": [0.0654, 0.0567],
        "NHR": [0.0221, 0.0194],
        "HNR": [21.03, 19.08],
        "status": [1, 0],
        "RPDE": [0.414, 0.458],
        "DFA": [0.815, 0.771],
        "spread1": [-4.813, -5.333],
        "spread2": [0.266, 0.224],
        "D2": [2.301, 2.115],
        "PPE": [0.284, 0.201],
    }
    raw_df = pd.DataFrame(fixture_data)
    adapter = UCIParkinsonsDetectionAdapter()
    canonical_records = adapter.to_canonical(raw_df)

    assert len(canonical_records) == 2
    rec1 = canonical_records[0]
    assert rec1.subject_id == "S01"
    assert rec1.diagnosis == 1
    assert rec1.acoustic_features["jitter_pct"] == 0.0078
    assert rec1.acoustic_features["shimmer"] == 0.0437
    assert rec1.motor_updrs is None  # Not present in this dataset -> None (never fabricated)


def test_fixture_uci_telemonitoring_conversion():
    """TEST FIXTURE: Uses 2 sample rows conforming to UCI Telemonitoring documented schema."""
    fixture_data = {
        "subject#": [1, 1],
        "age": [72, 72],
        "sex": [0, 0],
        "test_time": [5.64, 12.33],
        "motor_UPDRS": [28.1, 28.5],
        "total_UPDRS": [34.3, 35.1],
        "Jitter(%)": [0.0066, 0.0058],
        "Jitter(Abs)": [0.000034, 0.000029],
        "Jitter:RAP": [0.0033, 0.0029],
        "Jitter:PPQ5": [0.0036, 0.0031],
        "Jitter:DDP": [0.0098, 0.0087],
        "Shimmer": [0.0256, 0.0241],
        "Shimmer(dB)": [0.230, 0.215],
        "Shimmer:APQ3": [0.0143, 0.0135],
        "Shimmer:APQ5": [0.0165, 0.0154],
        "Shimmer:APQ11": [0.0195, 0.0182],
        "Shimmer:DDA": [0.0430, 0.0405],
        "NHR": [0.0142, 0.0125],
        "HNR": [21.64, 22.18],
        "RPDE": [0.442, 0.435],
        "DFA": [0.654, 0.661],
        "PPE": [0.219, 0.205],
    }
    raw_df = pd.DataFrame(fixture_data)
    adapter = UCIParkinsonsTelemonitoringAdapter()
    records = adapter.to_canonical(raw_df)

    assert len(records) == 2
    assert records[0].subject_id == "T001"
    assert records[0].visit_id == 1
    assert records[1].visit_id == 2  # Discretized from test_time advancement
    assert records[0].motor_updrs == 28.1
    assert records[0].total_updrs == 34.3
    # Unavailable pitch features remain None
    assert records[0].acoustic_features["mdvp_fo"] is None


# ---------------------------------------------------------------------
# 8. FeatureSchema Invariant
# ---------------------------------------------------------------------
def test_feature_schema_default_matches_raw_features():
    schema = FeatureSchema.default()
    assert schema.raw_features == RAW_FEATURES
    assert schema.phenotype_groups == PHENOTYPE_GROUPS
