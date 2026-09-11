"""
base.py
---------------------------------------------------------------
Abstract DatasetAdapter Base Class.
Defines the required contract for all dataset sources (Synthetic & Real UCI datasets).
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import pandas as pd

from src.data.schema import CanonicalRecord, FeatureAvailability, DatasetManifest
from src.data.validation import ValidationReport


class DatasetAdapter(ABC):
    """Abstract base class for all dataset adapters."""

    def __init__(
        self,
        dataset_id: str,
        name: str,
        source_url: str,
        research_role: str,
        license_info: str = "UCI Machine Learning Repository License / Open Access",
        expected_files: Optional[List[str]] = None,
        raw_dir: str = "",
        processed_dir: str = "",
    ):
        self.dataset_id = dataset_id
        self.name = name
        self.source_url = source_url
        self.research_role = research_role
        self.license_info = license_info
        self.expected_files = expected_files or []
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    @abstractmethod
    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        """Validates that expected source files exist and conform to schema."""
        pass

    @abstractmethod
    def load_raw(self, raw_dir: Optional[str] = None) -> pd.DataFrame:
        """Loads raw dataset files without modification into a DataFrame."""
        pass

    @abstractmethod
    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        """Converts raw data records into typed CanonicalRecord instances."""
        pass

    @abstractmethod
    def to_canonical_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to export canonical representation as a DataFrame."""
        pass

    @abstractmethod
    def get_feature_availability(self) -> Dict[str, FeatureAvailability]:
        """Returns feature availability map for this dataset."""
        pass

    @abstractmethod
    def get_available_task_types(self) -> List[str]:
        """Returns list of acoustic/clinical tasks captured in this dataset."""
        pass

    def has_longitudinal_data(self) -> bool:
        """Whether dataset contains genuine longitudinal clinical visits."""
        return False

    def has_updrs(self) -> bool:
        """Whether dataset contains UPDRS or motor scores."""
        return False

    def get_manifest(self) -> DatasetManifest:
        """Returns structured metadata manifest including current download/validation status."""
        v = self.validate_source()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            name=self.name,
            source_url=self.source_url,
            research_role=self.research_role,
            license_info=self.license_info,
            raw_path=self.raw_dir,
            processed_path=self.processed_dir,
            status=v.status,
            has_longitudinal_data=self.has_longitudinal_data(),
            has_updrs=self.has_updrs(),
            expected_files=self.expected_files,
            description=f"Adapter for {self.name} (Role: {self.research_role})",
        )
