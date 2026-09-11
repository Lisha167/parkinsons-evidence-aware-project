"""
synthetic.py
---------------------------------------------------------------
Adapter for the Synthetic Longitudinal Dataset.
Identifies dataset explicitly as 'synthetic' and wraps generate_synthetic_data.py.
"""
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.base import DatasetAdapter
from src.data.schema import CanonicalRecord, FeatureAvailability
from src.data.validation import ValidationReport
from src.features import RAW_FEATURES


class SyntheticAdapter(DatasetAdapter):
    """Adapter for the synthetic longitudinal dataset."""

    def __init__(self, data_path: Optional[str] = None):
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
        raw_path = data_path or os.path.join(default_dir, "patients_visits.csv")
        super().__init__(
            dataset_id="synthetic",
            name="Synthetic Longitudinal Parkinson's Voice Cohort",
            source_url="local://data/generate_synthetic_data.py",
            research_role="controlled_experiment",
            license_info="Internal Research Benchmark (Synthetic)",
            expected_files=["patients_visits.csv"],
            raw_dir=default_dir,
            processed_dir=default_dir,
        )
        self.data_file = raw_path

    def has_longitudinal_data(self) -> bool:
        return True

    def has_updrs(self) -> bool:
        return True

    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        path = self.data_file
        if raw_dir:
            path = os.path.join(raw_dir, "patients_visits.csv")
            
        if not os.path.exists(path):
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="not_downloaded",
                files_found=[],
                files_missing=["patients_visits.csv"],
                warnings=["Synthetic data file does not exist yet. Can be generated with data/generate_synthetic_data.py"],
            )

        try:
            df = pd.read_csv(path)
            req_cols = ["patient_id", "visit_id", "label"] + RAW_FEATURES
            missing_cols = [c for c in req_cols if c not in df.columns]
            found_cols = [c for c in req_cols if c in df.columns]
            n_sub = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
            missingness = {col: float(df[col].isna().mean()) for col in found_cols if df[col].isna().sum() > 0}
            dups = int(df.duplicated(subset=["patient_id", "visit_id"]).sum()) if "patient_id" in df.columns and "visit_id" in df.columns else 0

            is_valid = len(missing_cols) == 0
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=is_valid,
                status="ready" if is_valid else "invalid",
                files_found=[os.path.basename(path)],
                files_missing=[],
                columns_found=found_cols,
                columns_missing=missing_cols,
                n_records=len(df),
                n_subjects=n_sub,
                missingness=missingness,
                duplicates_count=dups,
            )
        except Exception as e:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="invalid",
                files_found=[os.path.basename(path)],
                type_errors=[str(e)],
            )

    def load_raw(self, raw_dir: Optional[str] = None) -> pd.DataFrame:
        path = self.data_file
        if raw_dir:
            path = os.path.join(raw_dir, "patients_visits.csv")
        if not os.path.exists(path):
            from data.generate_synthetic_data import generate
            df = generate(out_path=path)
            return df
        return pd.read_csv(path)

    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        records: List[CanonicalRecord] = []
        for _, row in raw_df.iterrows():
            acoustics = {f: (None if pd.isna(row.get(f)) else float(row[f])) for f in RAW_FEATURES if f in row}
            rec = CanonicalRecord(
                dataset_id=self.dataset_id,
                subject_id=str(row["patient_id"]),
                recording_id=f"{row['patient_id']}_v{row['visit_id']}",
                session_id=f"session_v{row['visit_id']}",
                visit_id=int(row["visit_id"]),
                timestamp=str(row.get("visit_date", "")),
                diagnosis=int(row["label"]) if "label" in row and not pd.isna(row["label"]) else None,
                task_type="sustained_vowel",
                recording_type="synthetic_simulation",
                acoustic_features=acoustics,
                motor_updrs=float(row["updrs_like"]) if "updrs_like" in row and not pd.isna(row["updrs_like"]) else None,
                total_updrs=float(row["updrs_like"]) if "updrs_like" in row and not pd.isna(row["updrs_like"]) else None,
                signal_quality=float(row["signal_quality"]) if "signal_quality" in row and not pd.isna(row["signal_quality"]) else None,
                demographic_metadata={"group": str(row.get("group", "unknown"))},
                source_metadata={
                    "known_change_event": int(row.get("known_change_event", 0)),
                    "change_type": str(row.get("change_type", "none")),
                },
            )
            records.append(rec)
        return records

    def to_canonical_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        records = self.to_canonical(raw_df)
        rows = []
        for r in records:
            d = {
                "dataset_id": r.dataset_id,
                "subject_id": r.subject_id,
                "recording_id": r.recording_id,
                "visit_id": r.visit_id,
                "diagnosis": r.diagnosis,
                "task_type": r.task_type,
                "signal_quality": r.signal_quality,
                "motor_updrs": r.motor_updrs,
                "total_updrs": r.total_updrs,
            }
            d.update(r.acoustic_features)
            rows.append(d)
        return pd.DataFrame(rows)

    def get_feature_availability(self) -> Dict[str, FeatureAvailability]:
        avail = {}
        for f in RAW_FEATURES:
            avail[f] = FeatureAvailability(
                feature_name=f,
                available=True,
                source="dataset_column",
                description=f"Synthetic simulation of {f}",
            )
        return avail

    def get_available_task_types(self) -> List[str]:
        return ["sustained_vowel"]
