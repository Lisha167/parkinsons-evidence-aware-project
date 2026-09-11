"""
uci_parkinsons_telemonitoring.py
---------------------------------------------------------------
Adapter for Dataset 2: UCI Parkinson's Telemonitoring (Tsanas et al., 2009).
URL: https://archive.ics.uci.edu/dataset/189/parkinson

Purpose: Longitudinal patient-specific change detection & UPDRS relationship.
Note: Genuine longitudinal time-series (repeated test_time days from baseline).
"""
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.base import DatasetAdapter
from src.data.schema import CanonicalRecord, FeatureAvailability
from src.data.validation import ValidationReport

UCI_TELEMONITORING_COLUMN_MAP = {
    "Jitter(%)": "jitter_pct",
    "Jitter(Abs)": "jitter_abs",
    "Jitter:RAP": "rap",
    "Jitter:PPQ5": "ppq",
    "Jitter:DDP": "ddp",
    "Shimmer": "shimmer",
    "Shimmer(dB)": "shimmer_db",
    "Shimmer:APQ3": "apq3",
    "Shimmer:APQ5": "apq5",
    "Shimmer:APQ11": "apq11",
    "Shimmer:DDA": "dda",
    "NHR": "nhr",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "PPE": "ppe",
}


class UCIParkinsonsTelemonitoringAdapter(DatasetAdapter):
    """Adapter for UCI Parkinson's Telemonitoring dataset."""

    def __init__(self, raw_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        r_dir = raw_dir or os.path.join(base_dir, "data", "raw", "uci_parkinsons_telemonitoring")
        p_dir = processed_dir or os.path.join(base_dir, "data", "processed", "uci_parkinsons_telemonitoring")
        super().__init__(
            dataset_id="uci_parkinsons_telemonitoring",
            name="UCI Parkinson's Telemonitoring",
            source_url="https://archive.ics.uci.edu/dataset/189/parkinson",
            research_role="longitudinal_change_detection",
            license_info="CC BY 4.0 / UCI Open Access (Tsanas et al., 2009)",
            expected_files=["parkinsons_updrs.data"],
            raw_dir=r_dir,
            processed_dir=p_dir,
        )

    def has_longitudinal_data(self) -> bool:
        return True  # True longitudinal tracking across ~6 months

    def has_updrs(self) -> bool:
        return True  # motor_UPDRS and total_UPDRS available

    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        target_dir = raw_dir or self.raw_dir
        expected_file = os.path.join(target_dir, "parkinsons_updrs.data")
        alt_file = os.path.join(target_dir, "parkinsons_updrs.csv")

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
                files_missing=["parkinsons_updrs.data (or parkinsons_updrs.csv)"],
                warnings=[
                    f"Dataset files not found in {target_dir}.",
                    "Download parkinsons_updrs.data from https://archive.ics.uci.edu/dataset/189/parkinson and place into data/raw/uci_parkinsons_telemonitoring/"
                ],
            )

        try:
            df = pd.read_csv(active_file)
            req_cols = ["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"] + list(UCI_TELEMONITORING_COLUMN_MAP.keys())
            missing_cols = [c for c in req_cols if c not in df.columns]
            found_cols = [c for c in df.columns]

            n_sub = int(df["subject#"].nunique()) if "subject#" in df.columns else 0
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
                n_subjects=n_sub,
                duplicates_count=0,
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
        expected_file = os.path.join(target_dir, "parkinsons_updrs.data")
        alt_file = os.path.join(target_dir, "parkinsons_updrs.csv")
        active = expected_file if os.path.exists(expected_file) else alt_file
        if not os.path.exists(active):
            raise FileNotFoundError(
                f"[{self.dataset_id}] Raw data file not found in {target_dir}. "
                "Please place 'parkinsons_updrs.data' from https://archive.ics.uci.edu/dataset/189/parkinson into the folder."
            )
        return pd.read_csv(active)

    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        records: List[CanonicalRecord] = []
        # Sort by subject and test_time to establish genuine visit order
        df_sorted = raw_df.sort_values(["subject#", "test_time"]).reset_index(drop=True)

        for sub_id, grp in df_sorted.groupby("subject#"):
            visit_counter = 1
            last_day = None
            for idx, row in grp.iterrows():
                day = float(row.get("test_time", 0.0))
                # If test_time advances by >= 1 full day, increment discrete visit index
                if last_day is None:
                    visit_counter = 1
                elif abs(day - last_day) >= 1.0:
                    visit_counter += 1
                last_day = day

                acoustics = {}
                for uci_col, can_col in UCI_TELEMONITORING_COLUMN_MAP.items():
                    val = row.get(uci_col)
                    acoustics[can_col] = None if pd.isna(val) else float(val)
                # Mark pitch floor/ceiling as None (not present in telemonitoring CSV)
                for f_missing in ["mdvp_fo", "mdvp_fhi", "mdvp_flo", "spread1", "spread2", "d2"]:
                    acoustics[f_missing] = None

                rec = CanonicalRecord(
                    dataset_id=self.dataset_id,
                    subject_id=f"T{int(sub_id):03d}",
                    recording_id=f"T{int(sub_id):03d}_rec{idx}",
                    session_id=f"session_day_{int(day)}",
                    visit_id=visit_counter,
                    timestamp=f"day_{day:.2f}",
                    diagnosis=1,  # All subjects in telemonitoring are PD patients tracked longitudinally
                    task_type="sustained_vowel",
                    recording_type="at_home_telemonitoring_device",
                    acoustic_features=acoustics,
                    motor_updrs=float(row["motor_UPDRS"]) if "motor_UPDRS" in row and not pd.isna(row["motor_UPDRS"]) else None,
                    total_updrs=float(row["total_UPDRS"]) if "total_UPDRS" in row and not pd.isna(row["total_UPDRS"]) else None,
                    signal_quality=None,
                    demographic_metadata={
                        "age": int(row["age"]) if "age" in row and not pd.isna(row["age"]) else None,
                        "sex": int(row["sex"]) if "sex" in row and not pd.isna(row["sex"]) else None,
                    },
                    source_metadata={"test_time": day},
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
                "timestamp": r.timestamp,
                "diagnosis": r.diagnosis,
                "task_type": r.task_type,
                "motor_updrs": r.motor_updrs,
                "total_updrs": r.total_updrs,
            }
            d.update(r.acoustic_features)
            rows.append(d)
        return pd.DataFrame(rows)

    def get_feature_availability(self) -> Dict[str, FeatureAvailability]:
        avail = {}
        for uci_col, can_col in UCI_TELEMONITORING_COLUMN_MAP.items():
            avail[can_col] = FeatureAvailability(
                feature_name=can_col,
                available=True,
                source="dataset_column",
                derived_from=uci_col,
                description=f"Direct mapped from UCI '{uci_col}'",
            )
        # Explicit unavailable fields
        unavail = ["mdvp_fo", "mdvp_fhi", "mdvp_flo", "spread1", "spread2", "d2", "signal_quality"]
        for u in unavail:
            avail[u] = FeatureAvailability(
                feature_name=u,
                available=False,
                source="unavailable",
                description="Not extracted in UCI Parkinson's Telemonitoring dataset",
            )
        # Motor & total UPDRS are available
        avail["motor_updrs"] = FeatureAvailability(feature_name="motor_updrs", available=True, source="dataset_column")
        avail["total_updrs"] = FeatureAvailability(feature_name="total_updrs", available=True, source="dataset_column")
        return avail

    def get_available_task_types(self) -> List[str]:
        return ["sustained_vowel"]
