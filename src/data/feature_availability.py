"""
feature_availability.py
---------------------------------------------------------------
Feature Availability System.
Explicitly records available, derived, and unavailable features per dataset.
Prevents downstream phenotype/evidence system from fabricating missing measurements.
"""
from typing import Dict, List, Optional, Any
from src.data.schema import FeatureAvailability
from src.data.adapters.synthetic import SyntheticAdapter
from src.data.adapters.uci_parkinsons_detection import UCIParkinsonsDetectionAdapter
from src.data.adapters.uci_parkinsons_telemonitoring import UCIParkinsonsTelemonitoringAdapter
from src.data.adapters.uci_parkinsons_multiple_speech import UCIParkinsonsMultipleSpeechAdapter
from src.data.adapters.uci_parkinsons_replicated import UCIParkinsonsReplicatedAdapter


def get_dataset_feature_availability(dataset_id: str) -> Dict[str, FeatureAvailability]:
    """Returns the feature availability map for a specific dataset."""
    adapters = {
        "synthetic": SyntheticAdapter,
        "uci_parkinsons_detection": UCIParkinsonsDetectionAdapter,
        "uci_parkinsons_telemonitoring": UCIParkinsonsTelemonitoringAdapter,
        "uci_parkinsons_multiple_speech": UCIParkinsonsMultipleSpeechAdapter,
        "uci_parkinsons_replicated": UCIParkinsonsReplicatedAdapter,
    }
    if dataset_id not in adapters:
        raise ValueError(f"Unknown dataset_id: '{dataset_id}'. Registered: {list(adapters.keys())}")
    
    adapter = adapters[dataset_id]()
    return adapter.get_feature_availability()


def check_phenotype_axis_coverage(dataset_id: str, phenotype_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Evaluates what proportion of features in each phenotype axis are available.
    
    Returns 'complete', 'partially_observed', or 'insufficient_evidence' per axis.
    """
    avail = get_dataset_feature_availability(dataset_id)
    coverage = {}

    for axis_name, features in phenotype_groups.items():
        present = [f for f in features if avail.get(f, FeatureAvailability(f, False)).available]
        missing = [f for f in features if not avail.get(f, FeatureAvailability(f, False)).available]
        pct = len(present) / len(features) if features else 0.0

        if pct == 1.0:
            status = "complete"
        elif pct >= 0.5:
            status = "partially_observed"
        else:
            status = "insufficient_evidence"

        coverage[axis_name] = {
            "status": status,
            "coverage_pct": round(pct, 3),
            "available_features": present,
            "missing_features": missing,
        }

    return coverage
