"""
splitting.py
---------------------------------------------------------------
Data Leakage Protection and Patient-Level Partitioning.
Ensures zero subject leakage across train/test and enforces strict
temporal isolation for longitudinal evaluation.
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def subject_level_split(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions records into train and test sets strictly at the subject level.
    Guarantees no records from a subject in train appear in test.
    """
    if subject_col not in df.columns:
        # Fallback to patient_id if present
        if "patient_id" in df.columns:
            subject_col = "patient_id"
        else:
            raise ValueError(f"Subject column '{subject_col}' not found in DataFrame.")

    unique_subjects = np.array(df[subject_col].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_test = max(1, int(len(unique_subjects) * test_size))
    test_subs = set(unique_subjects[:n_test])
    train_subs = set(unique_subjects[n_test:])

    train_df = df[df[subject_col].isin(train_subs)].reset_index(drop=True)
    test_df = df[df[subject_col].isin(test_subs)].reset_index(drop=True)

    validate_no_leakage(train_df, test_df, subject_col=subject_col)
    return train_df, test_df


def subject_level_kfold(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Subject-level K-Fold cross validation generator."""
    if subject_col not in df.columns:
        if "patient_id" in df.columns:
            subject_col = "patient_id"
        else:
            raise ValueError(f"Subject column '{subject_col}' not found in DataFrame.")

    unique_subjects = np.array(df[subject_col].unique(), dtype=object)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for train_idx, test_idx in kf.split(unique_subjects):
        train_subs = set(unique_subjects[train_idx])
        test_subs = set(unique_subjects[test_idx])
        train_df = df[df[subject_col].isin(train_subs)].reset_index(drop=True)
        test_df = df[df[subject_col].isin(test_subs)].reset_index(drop=True)
        validate_no_leakage(train_df, test_df, subject_col=subject_col)
        splits.append((train_df, test_df))

    return splits


def longitudinal_temporal_split(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    time_col: str = "visit_id",
    cutoff_time: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split for longitudinal data.
    Train uses only past observations (< cutoff_time); Test uses observations (>= cutoff_time).
    Prevents future observations from influencing baseline.
    """
    if subject_col not in df.columns:
        if "patient_id" in df.columns:
            subject_col = "patient_id"
    if time_col not in df.columns:
        if "test_time" in df.columns:
            time_col = "test_time"

    past_df = df[df[time_col] < cutoff_time].reset_index(drop=True)
    future_df = df[df[time_col] >= cutoff_time].reset_index(drop=True)
    return past_df, future_df


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, subject_col: str = "subject_id") -> bool:
    """Verifies intersection between train and test subjects is strictly empty."""
    if subject_col not in train_df.columns:
        if "patient_id" in train_df.columns:
            subject_col = "patient_id"
    
    train_subs = set(train_df[subject_col].unique())
    test_subs = set(test_df[subject_col].unique())
    overlap = train_subs.intersection(test_subs)
    if len(overlap) > 0:
        raise AssertionError(f"CRITICAL DATA LEAKAGE: {len(overlap)} subjects appear in both train and test sets: {list(overlap)[:5]}")
    return True
