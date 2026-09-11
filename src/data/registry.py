"""
registry.py
---------------------------------------------------------------
Central Dataset Registry & Provider.
Maintains known datasets, adapters, path configurations, and statuses.
"""
import os
from typing import Dict, List, Optional, Any
from src.data.base import DatasetAdapter
from src.data.schema import DatasetManifest
from src.data.validation import ValidationReport
from src.data.adapters.synthetic import SyntheticAdapter
from src.data.adapters.uci_parkinsons_detection import UCIParkinsonsDetectionAdapter
from src.data.adapters.uci_parkinsons_telemonitoring import UCIParkinsonsTelemonitoringAdapter
from src.data.adapters.uci_parkinsons_multiple_speech import UCIParkinsonsMultipleSpeechAdapter
from src.data.adapters.uci_parkinsons_replicated import UCIParkinsonsReplicatedAdapter


# Dataset Registry mapping ID -> Adapter Class
REGISTRY_MAP: Dict[str, type] = {
    "synthetic": SyntheticAdapter,
    "uci_parkinsons_detection": UCIParkinsonsDetectionAdapter,
    "uci_parkinsons_telemonitoring": UCIParkinsonsTelemonitoringAdapter,
    "uci_parkinsons_multiple_speech": UCIParkinsonsMultipleSpeechAdapter,
    "uci_parkinsons_replicated": UCIParkinsonsReplicatedAdapter,
}


class DatasetRegistry:
    """Central registry manager for all datasets."""

    @staticmethod
    def list_datasets() -> List[str]:
        """Returns all registered dataset IDs."""
        return list(REGISTRY_MAP.keys())

    @staticmethod
    def get_adapter(dataset_id: str, **kwargs) -> DatasetAdapter:
        """Instantiates and returns the adapter for the given dataset ID."""
        if dataset_id not in REGISTRY_MAP:
            raise KeyError(f"Dataset '{dataset_id}' not found in registry. Registered: {list(REGISTRY_MAP.keys())}")
        return REGISTRY_MAP[dataset_id](**kwargs)

    @classmethod
    def get_manifest(cls, dataset_id: str) -> DatasetManifest:
        """Returns the structured manifest for a specific dataset."""
        adapter = cls.get_adapter(dataset_id)
        return adapter.get_manifest()

    @classmethod
    def get_all_manifests(cls) -> Dict[str, DatasetManifest]:
        """Returns manifests for all registered datasets."""
        return {d_id: cls.get_manifest(d_id) for d_id in REGISTRY_MAP}

    @classmethod
    def validate_dataset(cls, dataset_id: str, raw_dir: Optional[str] = None) -> ValidationReport:
        """Runs validation checks for a registered dataset."""
        adapter = cls.get_adapter(dataset_id)
        return adapter.validate_source(raw_dir=raw_dir)

    @classmethod
    def validate_all(cls) -> Dict[str, ValidationReport]:
        """Runs validation checks across all registered datasets."""
        return {d_id: cls.validate_dataset(d_id) for d_id in REGISTRY_MAP}
