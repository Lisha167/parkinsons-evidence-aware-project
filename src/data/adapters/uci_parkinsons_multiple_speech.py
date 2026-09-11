"""
uci_parkinsons_multiple_speech.py
---------------------------------------------------------------
Adapter for Dataset 3: UCI Parkinson's Speech with Multiple Types of Sound Recordings (Sakar et al., 2013).
URL: https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings

Purpose: Multiple speech assessment types (sustained vowels, words, sentences).
Research role: Sequential assessment benchmarking and task comparison.
"""
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.base import DatasetAdapter
from src.data.schema import CanonicalRecord, FeatureAvailability
from src.data.validation import ValidationReport


class UCIParkinsonsMultipleSpeechAdapter(DatasetAdapter):
    """Adapter for UCI Parkinson's Speech with Multiple Sound Recordings."""

    def __init__(self, raw_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        r_dir = raw_dir or os.path.join(base_dir, "data", "raw", "uci_parkinsons_multiple_speech")
        p_dir = processed_dir or os.path.join(base_dir, "data", "processed", "uci_parkinsons_multiple_speech")
        super().__init__(
            dataset_id="uci_parkinsons_multiple_speech",
            name="UCI Parkinson's Speech with Multiple Types of Sound Recordings",
            source_url="https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings",
            research_role="sequential_speech_assessment",
            license_info="CC BY 4.0 / UCI Open Access (Sakar et al., 2013)",
            expected_files=["train_data.txt", "test_data.txt"],
            raw_dir=r_dir,
            processed_dir=p_dir,
        )

    def has_longitudinal_data(self) -> bool:
        return False  # Multi-task cross-sectional recording session (not longitudinal progression)

    def has_updrs(self) -> bool:
        return True  # Includes clinical UPDRS ratings

    def validate_source(self, raw_dir: Optional[str] = None) -> ValidationReport:
        target_dir = raw_dir or self.raw_dir
        expected = ["train_data.txt", "test_data.txt"]
        found = [f for f in expected if os.path.exists(os.path.join(target_dir, f))]
        missing = [f for f in expected if not os.path.exists(os.path.join(target_dir, f))]

        if len(found) == 0:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="not_downloaded",
                files_found=[],
                files_missing=missing,
                warnings=[
                    f"Dataset files not found in {target_dir}.",
                    "Download train_data.txt and test_data.txt from https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings and place into data/raw/uci_parkinsons_multiple_speech/"
                ],
            )

        # Partial download
        if len(missing) > 0:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="missing_files",
                files_found=found,
                files_missing=missing,
                warnings=["Partial dataset found. Both train_data.txt and test_data.txt expected."],
            )

        try:
            # Check train file header/contents
            train_path = os.path.join(target_dir, "train_data.txt")
            df = pd.read_csv(train_path, header=None)
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=True,
                status="validated",
                files_found=found,
                files_missing=[],
                n_records=len(df),
                n_subjects=int(df[0].nunique()) if len(df.columns) > 0 else 0,
            )
        except Exception as e:
            return ValidationReport(
                dataset_id=self.dataset_id,
                is_valid=False,
                status="invalid",
                files_found=found,
                type_errors=[str(e)],
            )

    def load_raw(self, raw_dir: Optional[str] = None) -> pd.DataFrame:
        target_dir = raw_dir or self.raw_dir
        train_path = os.path.join(target_dir, "train_data.txt")
        test_path = os.path.join(target_dir, "test_data.txt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"[{self.dataset_id}] Raw data files not found in {target_dir}. "
                "Download train_data.txt and test_data.txt from UCI repository."
            )
        dfs = []
        if os.path.exists(train_path):
            d1 = pd.read_csv(train_path, header=None)
            d1["split_source"] = "train"
            dfs.append(d1)
        if os.path.exists(test_path):
            d2 = pd.read_csv(test_path, header=None)
            d2["split_source"] = "test"
            dfs.append(d2)
        return pd.concat(dfs, ignore_index=True)

    def to_canonical(self, raw_df: pd.DataFrame) -> List[CanonicalRecord]:
        records: List[CanonicalRecord] = []
        # Column 0: Subject ID
        # Columns 1-26: Acoustic biomarkers (Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE, etc.)
        # Last column before split_source: class label (0/1)
        for idx, row in raw_df.iterrows():
            sub_id = f"MS_{int(row[0])}"
            label_col = len(row) - 2 if "split_source" in row else len(row) - 1
            diag = int(row[label_col]) if not pd.isna(row[label_col]) else None

            # Multi-task assignment: In Sakar et al., 26 recordings per patient:
            # 1-3: sustained vowel /a/, 4-6: vowel /o/, 7-9: vowel /u/, 10-19: numbers, 20-26: short sentences
            rec_idx = idx % 26
            if rec_idx < 9:
                task = "sustained_vowel"
            elif rec_idx < 19:
                task = "reading_words"
            else:
                task = "reading_passage"

            acoustics = {
                "jitter_pct": float(row[1]) if len(row) > 1 and not pd.isna(row[1]) else None,
                "jitter_abs": float(row[2]) if len(row) > 2 and not pd.isna(row[2]) else None,
                "rap": float(row[3]) if len(row) > 3 and not pd.isna(row[3]) else None,
                "ppq": float(row[4]) if len(row) > 4 and not pd.isna(row[4]) else None,
                "ddp": float(row[5]) if len(row) > 5 and not pd.isna(row[5]) else None,
                "shimmer": float(row[6]) if len(row) > 6 and not pd.isna(row[6]) else None,
                "shimmer_db": float(row[7]) if len(row) > 7 and not pd.isna(row[7]) else None,
                "apq3": float(row[8]) if len(row) > 8 and not pd.isna(row[8]) else None,
                "apq5": float(row[9]) if len(row) > 9 and not pd.isna(row[9]) else None,
                "apq11": float(row[10]) if len(row) > 10 and not pd.isna(row[10]) else None,
                "dda": float(row[11]) if len(row) > 11 and not pd.isna(row[11]) else None,
                "nhr": float(row[12]) if len(row) > 12 and not pd.isna(row[12]) else None,
                "hnr": float(row[13]) if len(row) > 13 and not pd.isna(row[13]) else None,
                "rpde": float(row[14]) if len(row) > 14 and not pd.isna(row[14]) else None,
                "dfa": float(row[15]) if len(row) > 15 and not pd.isna(row[15]) else None,
                "ppe": float(row[16]) if len(row) > 16 and not pd.isna(row[16]) else None,
            }

            rec = CanonicalRecord(
                dataset_id=self.dataset_id,
                subject_id=sub_id,
                recording_id=f"{sub_id}_task{rec_idx}",
                session_id=None,
                visit_id=None,
                timestamp=None,
                diagnosis=diag,
                task_type=task,
                recording_type="microphone_acoustic",
                acoustic_features=acoustics,
                source_metadata={"recording_sequence_in_session": rec_idx},
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
        standard_cols = ["jitter_pct", "jitter_abs", "rap", "ppq", "ddp", "shimmer", "shimmer_db",
                         "apq3", "apq5", "apq11", "dda", "nhr", "hnr", "rpde", "dfa", "ppe"]
        for c in standard_cols:
            avail[c] = FeatureAvailability(feature_name=c, available=True, source="dataset_column")
        for u in ["mdvp_fo", "mdvp_fhi", "mdvp_flo", "spread1", "spread2", "d2", "signal_quality"]:
            avail[u] = FeatureAvailability(feature_name=u, available=False, source="unavailable")
        return avail

    def get_available_task_types(self) -> List[str]:
        return ["sustained_vowel", "reading_words", "reading_passage"]
