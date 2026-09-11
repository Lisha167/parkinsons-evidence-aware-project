"""
schema.py
---------------------------------------------------------------
Canonical Data Schema & Manifest Types for Parkinson's Voice Assessment.
Supports multi-dataset representations (Synthetic, UCI Detection, UCI Telemonitoring,
UCI Multiple Speech, UCI Replicated Acoustic) with optional fields.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class CanonicalRecord:
    """Canonical internal record representation across datasets.
    
    Fields are optional where a dataset does not provide them.
    Missing values are kept as None and NEVER fabricated.
    """
    dataset_id: str
    subject_id: str
    recording_id: Optional[str] = None
    session_id: Optional[str] = None
    visit_id: Optional[int] = None
    timestamp: Optional[str] = None
    diagnosis: Optional[int] = None  # 0=healthy, 1=PD, None=unlabeled/unknown
    task_type: Optional[str] = None  # e.g. "sustained_vowel", "reading", "ddk", etc.
    recording_type: Optional[str] = None
    acoustic_features: Dict[str, Optional[float]] = field(default_factory=dict)
    clinical_measurements: Dict[str, Optional[float]] = field(default_factory=dict)
    motor_updrs: Optional[float] = None
    total_updrs: Optional[float] = None
    signal_quality: Optional[float] = None
    demographic_metadata: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recording:
    """Single voice recording instance."""
    recording_id: str
    task_type: Optional[str] = None
    acoustic_features: Dict[str, Optional[float]] = field(default_factory=dict)
    signal_quality: Optional[float] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Group of recordings in a single session or clinical visit."""
    session_id: str
    recordings: List[Recording] = field(default_factory=list)
    clinical_measurements: Dict[str, Optional[float]] = field(default_factory=dict)
    motor_updrs: Optional[float] = None
    total_updrs: Optional[float] = None
    timestamp: Optional[str] = None
    is_longitudinal_visit: bool = False  # True ONLY if genuine clinical longitudinal visit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatientRecord:
    """Subject/Patient entity holding sessions and metadata."""
    dataset_id: str
    subject_id: str
    diagnosis: Optional[int] = None
    sessions: List[Session] = field(default_factory=list)
    demographic_metadata: Dict[str, Any] = field(default_factory=dict)
    has_longitudinal_data: bool = False
    has_updrs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureAvailability:
    """Explicit metadata describing availability and provenance of a feature."""
    feature_name: str
    available: bool
    source: str = "unknown"  # "dataset_column", "derived", "unavailable"
    derived_from: Optional[str] = None
    unit: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    """Metadata describing a dataset and its operational status."""
    dataset_id: str
    name: str
    source_url: str
    research_role: str  # "screening", "longitudinal_change_detection", "sequential_speech_assessment", "within_subject_variability", "controlled_experiment"
    license_info: str
    raw_path: str
    processed_path: str
    status: str  # "not_downloaded", "downloaded", "validated", "preprocessed", "ready"
    has_longitudinal_data: bool
    has_updrs: bool
    expected_files: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssessmentCapability:
    """Assessment modalities supported by datasets."""
    assessment_id: str
    name: str
    modality: str
    supported_datasets: List[str]
    required_features: List[str]
    availability_status: str  # "available", "unavailable", "simulated"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
