"""
validation.py
---------------------------------------------------------------
Data validation routines and reporting for raw/processed datasets.
Ensures zero synthetic values inserted, checks missingness, duplicates,
and schema adherence.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import os
import pandas as pd


@dataclass
class ValidationReport:
    """Structured report returned after validating a dataset directory or file."""
    dataset_id: str
    is_valid: bool
    status: str  # "ready", "validated", "not_downloaded", "missing_files", "invalid"
    files_found: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    columns_found: List[str] = field(default_factory=list)
    columns_missing: List[str] = field(default_factory=list)
    n_records: int = 0
    n_subjects: int = 0
    missingness: Dict[str, float] = field(default_factory=dict)
    duplicates_count: int = 0
    type_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            f"Dataset: {self.dataset_id}",
            f"Valid: {self.is_valid} (Status: {self.status})",
            f"Files Found: {len(self.files_found)} | Missing: {len(self.files_missing)}",
            f"Records: {self.n_records} | Subjects: {self.n_subjects}",
            f"Duplicates: {self.duplicates_count}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.type_errors:
            lines.append("Errors:")
            for e in self.type_errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)
