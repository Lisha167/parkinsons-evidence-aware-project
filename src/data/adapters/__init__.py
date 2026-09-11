"""
adapters/__init__.py
---------------------------------------------------------------
Dataset Adapter package exports.
"""
from src.data.adapters.synthetic import SyntheticAdapter
from src.data.adapters.uci_parkinsons_detection import UCIParkinsonsDetectionAdapter
from src.data.adapters.uci_parkinsons_telemonitoring import UCIParkinsonsTelemonitoringAdapter
from src.data.adapters.uci_parkinsons_multiple_speech import UCIParkinsonsMultipleSpeechAdapter
from src.data.adapters.uci_parkinsons_replicated import UCIParkinsonsReplicatedAdapter

__all__ = [
    "SyntheticAdapter",
    "UCIParkinsonsDetectionAdapter",
    "UCIParkinsonsTelemonitoringAdapter",
    "UCIParkinsonsMultipleSpeechAdapter",
    "UCIParkinsonsReplicatedAdapter",
]
