"""
assessment_capability.py
---------------------------------------------------------------
Assessment Capability Registry for Next-Best Assessment Selection.
Maintains which speech/clinical assessment modalities are supported by each dataset.
Prevents the next-best assessment engine from proposing tests unavailable in the active dataset.
"""
from typing import Dict, List, Optional
from src.data.schema import AssessmentCapability

ASSESSMENT_CAPABILITIES: Dict[str, AssessmentCapability] = {
    "repeat_sustained_vowel": AssessmentCapability(
        assessment_id="repeat_sustained_vowel",
        name="Repeat Sustained Vowel",
        modality="acoustic_phonation",
        supported_datasets=["synthetic", "uci_parkinsons_detection", "uci_parkinsons_multiple_speech", "uci_parkinsons_replicated"],
        required_features=["jitter_pct", "shimmer", "hnr"],
        availability_status="available",
        description="Repeated /a/ vowel phonation evaluating phonatory stability.",
    ),
    "reading_passage_task": AssessmentCapability(
        assessment_id="reading_passage_task",
        name="Standardized Reading Passage",
        modality="acoustic_continuous_speech",
        supported_datasets=["synthetic", "uci_parkinsons_multiple_speech"],
        required_features=["prosodic_variability", "speech_rate"],
        availability_status="available",
        description="Connected speech task measuring articulation rate and prosodic variation.",
    ),
    "diadochokinetic_task": AssessmentCapability(
        assessment_id="diadochokinetic_task",
        name="Diadochokinetic (DDK) Speech Task",
        modality="acoustic_articulatory",
        supported_datasets=["synthetic"],
        required_features=["ddk_syllable_rate", "ddk_regularity"],
        availability_status="simulated",
        description="Rapid syllable repetition (/pa-ta-ka/) capturing articulatory agility.",
    ),
    "extended_home_monitoring": AssessmentCapability(
        assessment_id="extended_home_monitoring",
        name="7-Day Longitudinal Home Voice Diary",
        modality="longitudinal_ecological",
        supported_datasets=["synthetic", "uci_parkinsons_telemonitoring"],
        required_features=["test_time", "jitter_pct", "shimmer"],
        availability_status="available",
        description="Repeated home telemonitoring measurements over weeks/months.",
    ),
    "accelerometer_gait_check": AssessmentCapability(
        assessment_id="accelerometer_gait_check",
        name="Wearable Inertial Gait & Tremor Check",
        modality="multimodal_inertial",
        supported_datasets=["synthetic"],
        required_features=["resting_tremor_power_hz", "stride_cv"],
        availability_status="simulated",
        description="Cross-modality inertial sensor evaluation.",
    ),
    "clinical_updrs_exam": AssessmentCapability(
        assessment_id="clinical_updrs_exam",
        name="Clinician UPDRS-III Motor Examination",
        modality="clinical_gold_standard",
        supported_datasets=["synthetic", "uci_parkinsons_telemonitoring", "uci_parkinsons_multiple_speech"],
        required_features=["motor_updrs"],
        availability_status="available",
        description="MDS-UPDRS clinician motor assessment.",
    ),
}


def is_assessment_available(assessment_id: str, dataset_id: str) -> bool:
    """Checks whether a specific assessment is supported by the dataset."""
    cap = ASSESSMENT_CAPABILITIES.get(assessment_id)
    if cap is None:
        return False
    return dataset_id in cap.supported_datasets


def get_available_assessments_for_dataset(dataset_id: str) -> List[AssessmentCapability]:
    """Returns all assessment capabilities supported by the dataset."""
    return [cap for cap in ASSESSMENT_CAPABILITIES.values() if dataset_id in cap.supported_datasets]
