"""
uci_parkinsons_replicated.py
---------------------------------------------------------------
Adapter for Dataset 4: UCI Parkinson Dataset with Replicated Acoustic Features (Naranjo et al., 2016).
URL: https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features

Purpose: Repeated measurements from same subjects (within-person variability analysis).
Research role: Baseline variability estimation & test-retest replication reliability.
"""
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.base import DatasetAdapter
from src.data.schema import CanonicalRecord, FeatureAvailability
from src.data.validation import ValidationReport


class UCIParkinsonsReplicatedAdapter(DatasetAdapter):
    """Adapter for UCI Parkinson Dataset with Replicated Acoustic Features."""

    def __init__(self, raw_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        r_dir = raw_dir or os.path.join(base_dir, "data", "raw", "uci_parkinsons_replicated")
        p_dir = processed_dir or os.path.join(base_dir, "data", "processed", "uci_parkinsons_replicated")
        super().__init__(
            dataset_id="uci_parkinsons_replicated",
            name="UCI Parkinson Dataset with Replicated Acoustic Features",
            source_url="https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features",
            research_role="within_subject_variability",
            license_info="CC BY 4.0 / UCI Open Access (Naranjo et al., 2016)",
            expected_files=["ReplicatedAcousticFeatures-ParkinsonDatabase.csv"],
            raw_dir=r_dir,
            processed_dir=p_dir,
        )

    def has_longitudinal_data(self) -> bool:
        return False  # Replicated within-session phonations, not time-series visits

    def has_updrs(self) -> bool:
        return False

    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        target_dir = raw_dir or self.raw_dir
        expected_file = os.path.join(target_dir, "ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
        alt_file = os.path.join(target_dir, "replicated_acoustic_features.csv")

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
                files_missing=["ReplicatedAcousticFeatures-ParkinsonDatabase.csv"],
                warnings=[
                    f"Dataset files not found in {target_dir}.",
                    "Download ReplicatedAcousticFeatures-ParkinsonDatabase.csv from https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features and place into data/raw/uci_parkinsons_replicated/"
                ],
            )

        try:
            df = pd.read_csv(active_file)
            req_cols = ["ID", "Status", "Recording"]
            missing_cols = [c for c in req_cols if c not in df.columns]
            found_cols = [c for c in df.columns]

            n_sub = int(df["ID"].nunique()) if "ID" in df.columns else 0
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
        expected_file = os.path.join(target_dir, "ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
        alt_file = os.path.join(target_dir, "replicated_acoustic_features.csv")
        active = expected_file if os.path.exists(expected_file) else alt_file
        if not os.path.exists(active):
            raise FileNotFoundError(
                f"[{self.dataset_id}] Raw data file not found in {target_dir}. "
                "Download ReplicatedAcousticFeatures-ParkinsonDatabase.csv from UCI repository."
            )
        return pd.read_csv(active)

    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        records: List[CanonicalRecord] = []
        for _, row in raw_df.iterrows():
            sub_id = f"R_{int(row['ID'])}" if "ID" in row and not pd.isna(row['ID']) else "R_unknown"
            rec_num = int(row.get("Recording", 1))
            diag = int(row["Status"]) if "Status" in row and not pd.isna(row["Status"]) else None

            # Extract available standard features
            acoustics = {
                "jitter_pct": float(row["Jitter_rel"]) if "Jitter_rel" in row and not pd.isna(row["Jitter_rel"]) else None,
                "jitter_abs": float(row["Jitter_abs"]) if "Jitter_abs" in row and not pd.isna(row["Jitter_abs"]) else None,
                "shimmer": float(row["Shimmer_rel"]) if "Shimmer_rel" in row and not pd.isna(row["Shimmer_rel"]) else None,
                "shimmer_db": float(row["Shimmer_dB"]) if "Shimmer_dB" in row and not pd.isna(row["Shimmer_dB"]) else None,
                "nhr": float(row["NHR"]) if "NHR" in row and not pd.isna(row["NHR"]) else None,
                "hnr": float(row["HNR"]) if "HNR" in row and not pd.isna(row["HNR"]) else None,
                "rpde": float(row["RPDE"]) if "RPDE" in row and not pd.isna(row["RPDE"]) else None,
                "dfa": float(row["DFA"]) if "DFA" in row and not pd.isna(row["DFA"]) else None,
                "ppe": float(row["PPE"]) if "PPE" in row and not pd.isna(row["PPE"]) else None,
            }

            rec = CanonicalRecord(
                dataset_id=self.dataset_id,
                subject_id=sub_id,
                recording_id=f"{sub_id}_rep{rec_num}",
                session_id=None,
                visit_id=None,  # Not clinical visits - replicated test-retest trials
                timestamp=None,
                diagnosis=diag,
                task_type="sustained_vowel",
                recording_type="microphone_replicated_phonation",
                acoustic_features=acoustics,
                source_metadata={"repetition_number": rec_num},
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
                "diagnosis": r.diagnosis,
                "task_type": r.task_type,
            }
            d.update(r.acoustic_features)
            rows.append(d)
        return pd.DataFrame(rows)

    def get_feature_availability(self) -> Dict[str, FeatureAvailability]:
        avail = {}
        standard = ["jitter_pct", "jitter_abs", "shimmer", "shimmer_db", "nhr", "hnr", "rpde", "dfa", "ppe"]
        for s in standard:
            avail[s] = FeatureAvailability(feature_name=s, available=True, source="dataset_column")
        for u in ["mdvp_fo", "mdvp_fhi", "mdvp_flo", "rap", "ppq", "ddp", "apq3", "apq5", "apq11", "dda", "spread1", "spread2", "d2", "signal_quality"]:
            avail[u] = FeatureAvailability(feature_name=u, available=False, source="unavailable")
        return avail

    def get_available_task_types(self) -> List[str]:
        return ["sustained_vowel"]
