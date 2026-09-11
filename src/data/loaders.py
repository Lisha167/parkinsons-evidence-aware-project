"""
loaders.py
---------------------------------------------------------------
Unified Data Loader functions.
Provides standard entrypoints for loading raw or canonical datasets.
"""
from typing import List, Optional, Union
import pandas as pd
from src.data.registry import DatasetRegistry
from src.data.schema import CanonicalRecord


def load_canonical_records(dataset_id: str, raw_dir: Optional[str] = None) -> List[CanonicalRecord]:
    """Loads a dataset through its adapter and returns canonical records."""
    adapter = DatasetRegistry.get_adapter(dataset_id, raw_dir=raw_dir) if raw_dir else DatasetRegistry.get_adapter(dataset_id)
    raw_df = adapter.load_raw()
    return adapter.to_canonical(raw_df)


def load_canonical_df(dataset_id: str, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads a dataset through its adapter and returns a canonical DataFrame."""
    adapter = DatasetRegistry.get_adapter(dataset_id, raw_dir=raw_dir) if raw_dir else DatasetRegistry.get_adapter(dataset_id)
    raw_df = adapter.load_raw()
    return adapter.to_canonical_dataframe(raw_df)


def load_raw_dataset(dataset_id: str, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads raw dataset without schema conversion."""
    adapter = DatasetRegistry.get_adapter(dataset_id, raw_dir=raw_dir) if raw_dir else DatasetRegistry.get_adapter(dataset_id)
    return adapter.load_raw()
