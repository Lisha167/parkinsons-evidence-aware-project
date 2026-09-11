"""
uci_parkinsons_detection.py
---------------------------------------------------------------
Adapter for Dataset 1: UCI Parkinson's Disease Detection (Little et al., 2007).
URL: https://archive.ics.uci.edu/dataset/174/parkinsons

Purpose: Baseline acoustic screening (PD vs Healthy).
Note: Cross-sectional, no true longitudinal visits.
"""
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.base import DatasetAdapter
from src.data.schema import CanonicalRecord, FeatureAvailability
from src.data.validation import ValidationReport

# UCI Detection dataset raw column mapping to internal acoustic feature names
UCI_DETECTION_COLUMN_MAP = {
    "MDVP:Fo(Hz)": "mdvp_fo",
    "MDVP:Fhi(Hz)": "mdvp_fhi",
    "MDVP:Flo(Hz)": "mdvp_flo",
    "MDVP:Jitter(%)": "jitter_pct",
    "MDVP:Jitter(Abs)": "jitter_abs",
    "MDVP:RAP": "rap",
    "MDVP:PPQ": "ppq",
    "Jitter:DDP": "ddp",
    "MDVP:Shimmer": "shimmer",
    "MDVP:Shimmer(dB)": "shimmer_db",
    "Shimmer:APQ3": "apq3",
    "Shimmer:APQ5": "apq5",
    "MDVP:APQ": "apq11",
    "Shimmer:DDA": "dda",
    "NHR": "nhr",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "spread1": "spread1",
    "spread2": "spread2",
    "D2": "d2",
    "PPE": "ppe",
}


class UCIParkinsonsDetectionAdapter(DatasetAdapter):
    """Adapter for UCI Parkinson's Disease Detection dataset."""

    def __init__(self, raw_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        r_dir = raw_dir or os.path.join(base_dir, "data", "raw", "uci_parkinsons_detection")
        p_dir = processed_dir or os.path.join(base_dir, "data", "processed", "uci_parkinsons_detection")
        super().__init__(
            dataset_id="uci_parkinsons_detection",
            name="UCI Parkinson's Disease Detection",
            source_url="https://archive.ics.uci.edu/dataset/174/parkinsons",
            research_role="screening",
            license_info="CC BY 4.0 / UCI Open Access (Little et al., 2007)",
            expected_files=["parkinsons.data"],
            raw_dir=r_dir,
            processed_dir=p_dir,
        )

    def has_longitudinal_data(self) -> bool:
        return False  # Cross-sectional screening cohort

    def has_updrs(self) -> bool:
        return False

    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        target_dir = raw_dir or self.raw_dir
        expected_file = os.path.join(target_dir, "parkinsons.data")
        alt_file = os.path.join(target_dir, "parkinsons.csv")

        active_file = None
        if os.path.exists(expected_file):
            active_file = expected_file
        elif os.path.exists(alt_file):
            active_file = alt_file

        if active_file is None:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="not_downloaded",
                files_found=[],
                files_missing=["parkinsons.data (or parkinsons.csv)"],
                warnings=[
                    f"Dataset files not found in {target_dir}.",
                    "Download parkinsons.data from https://archive.ics.uci.edu/dataset/174/parkinsons and place into data/raw/uci_parkinsons_detection/"
                ],
            )

        try:
            df = pd.read_csv(active_file)
            missing_cols = [c for c in list(UCI_DETECTION_COLUMN_MAP.keys()) + ["name", "status"] if c not in df.columns]
            found_cols = [c for c in df.columns]
            
            # Subject extraction from 'name' column: e.g. phon_R01_S01_1 -> S01
            subjects = set()
            if "name" in df.columns:
                for n in df["name"].dropna():
                    parts = str(n).split("_")
                    sub = parts[2] if len(parts) >= 3 else parts[0]
                    subjects.add(sub)
            
            is_valid = len(missing_cols) == 0
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=is_valid,
                status="validated" if is_valid else "invalid",
                files_found=[os.path.basename(active_file)],
                files_missing=[],
                columns_found=found_cols,
                columns_missing=missing_cols,
                n_records=len(df),
                n_subjects=len(subjects),
                duplicates_count=int(df.duplicated(subset=["name"]).sum()) if "name" in df.columns else 0,
            )
        except Exception as e:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="invalid",
                files_found=[os.path.basename(active_file)],
                type_errors=[str(e)],
            )

    def load_raw(self, raw_dir: Optional[str] = None) -> pd.DataFrame:
        target_dir = raw_dir or self.raw_dir
        expected_file = os.path.join(target_dir, "parkinsons.data")
        alt_file = os.path.join(target_dir, "parkinsons.csv")
        active = expected_file if os.path.exists(expected_file) else alt_file
        if not os.path.exists(active):
            raise FileNotFoundError(
                f"[{self.dataset_id}] Raw data file not found in {target_dir}. "
                "Please place 'parkinsons.data' from https://archive.ics.uci.edu/dataset/174/parkinsons into the folder."
            )
        return pd.read_csv(active)

    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        records: List[CanonicalRecord] = []
        for _, row in raw_df.iterrows():
            name_val = str(row.get("name", "unknown"))
            # e.g., phon_R01_S01_1 -> subject 'S01', recording '1'
            parts = name_val.split("_")
            sub_id = parts[2] if len(parts) >= 3 else name_val
            rec_id = name_val

            acoustics = {}
            for uci_col, can_col in UCI_DETECTION_COLUMN_MAP.items():
                val = row.get(uci_col)
                acoustics[can_col] = None if pd.isna(val) else float(val)

            diagnosis = int(row["status"]) if "status" in row and not pd.isna(row["status"]) else None

            rec = CanonicalRecord(
                dataset_id=self.dataset_id,
                subject_id=sub_id,
                recording_id=rec_id,
                session_id=None,
                visit_id=None,  # No genuine longitudinal visit structure
                timestamp=None,
                diagnosis=diagnosis,
                task_type="sustained_vowel",
                recording_type="microphone_acoustic",
                acoustic_features=acoustics,
                clinical_measurements={},
                motor_updrs=None,
                total_updrs=None,
                signal_quality=None,  # Dataset does not provide SNR proxy directly
                demographic_metadata={},
                source_metadata={"raw_name": name_val},
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
            }
            d.update(r.acoustic_features)
            rows.append(d)
        return pd.DataFrame(rows)

    def get_feature_availability(self) -> Dict[str, FeatureAvailability]:
        avail = {}
        for uci_col, can_col in UCI_DETECTION_COLUMN_MAP.items():
            avail[can_col] = FeatureAvailability(
                feature_name=can_col,
                available=True,
                source="dataset_column",
                derived_from=uci_col,
                description=f"Direct mapped from UCI '{uci_col}'",
            )
        # Features not present in this dataset
        unavail = ["signal_quality", "motor_updrs", "total_updrs", "visit_id"]
        for u in unavail:
            avail[u] = FeatureAvailability(
                feature_name=u,
                available=False,
                source="unavailable",
                description="Not collected in UCI Parkinson's Detection",
            )
        return avail

    def get_available_task_types(self) -> List[str]:
        return ["sustained_vowel"]
