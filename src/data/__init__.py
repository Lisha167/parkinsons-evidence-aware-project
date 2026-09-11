"""
src/data/__init__.py
---------------------------------------------------------------
Data Abstraction Layer package exports.
"""
from src.data.schema import (
    CanonicalRecord,
    Recording,
    Session,
    PatientRecord,
    FeatureAvailability,
    DatasetManifest,
    AssessmentCapability,
)
from src.data.base import DatasetAdapter
from src.data.validation import ValidationReport
from src.data.registry import DatasetRegistry
from src.data.loaders import load_canonical_records, load_canonical_df, load_raw_dataset
from src.data.feature_availability import get_dataset_feature_availability, check_phenotype_axis_coverage
from src.data.assessment_capability import (
    ASSESSMENT_CAPABILITIES,
    is_assessment_available,
    get_available_assessments_for_dataset,
)
from src.data.splitting import (
    subject_level_split,
    subject_level_kfold,
    longitudinal_temporal_split,
    validate_no_leakage,
)

__all__ = [
    "CanonicalRecord",
    "Recording",
    "Session",
    "PatientRecord",
    "FeatureAvailability",
    "DatasetManifest",
    "AssessmentCapability",
    "DatasetAdapter",
    "ValidationReport",
    "DatasetRegistry",
    "load_canonical_records",
    "load_canonical_df",
    "load_raw_dataset",
    "get_dataset_feature_availability",
    "check_phenotype_axis_coverage",
    "ASSESSMENT_CAPABILITIES",
    "is_assessment_available",
    "get_available_assessments_for_dataset",
    "subject_level_split",
    "subject_level_kfold",
    "longitudinal_temporal_split",
    "validate_no_leakage",
]
