"""
test_pipeline.py
---------------------------------------------------------------
Pipeline smoke and integration tests for Parkinson's Voice Assessment.
"""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.generate_synthetic_data import generate
from src.features import add_phenotype_columns, compute_signal_quality_metrics
from src.models import train_and_evaluate, VoiceEnsemble
from src.agents.orchestrator import DecisionOrchestrator
from src.evaluation import discrimination_and_calibration, selective_risk_curve, change_detection_metrics


def test_data_generation():
    df = generate(n_healthy=5, n_pd=5, scenario="moderate", seed=1)
    assert len(df) > 0
    assert set(["patient_id", "visit_id", "label", "known_change_event"]).issubset(df.columns)


def test_phenotype_and_signal_quality():
    df = generate(n_healthy=5, n_pd=5, seed=1)
    df2 = add_phenotype_columns(df)
    assert any(c.startswith("phenotype_") for c in df2.columns)
    
    sq = compute_signal_quality_metrics(df2.iloc[0])
    assert 0.0 <= sq.overall_quality <= 1.0
    assert 0.0 <= sq.missingness_rate <= 1.0


def test_train_and_evaluate():
    df = add_phenotype_columns(generate(n_healthy=10, n_pd=10, seed=1))
    ens, metrics, train_df, test_df = train_and_evaluate(df, seed=1)
    assert "auc" in metrics
    assert "expected_calibration_error" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_orchestrator_end_to_end_no_llm():
    df = add_phenotype_columns(generate(n_healthy=10, n_pd=10, seed=1))
    ens, metrics, train_df, test_df = train_and_evaluate(df, seed=1)
    pid = df[df["label"] == 1]["patient_id"].iloc[0]
    history = df[df["patient_id"] == pid].sort_values("visit_id").reset_index(drop=True)
    orch = DecisionOrchestrator(ens, use_llm=False)
    result = orch.run_case(history, int(history["visit_id"].max()))
    
    assert result["consensus"]["final_action"] in (
        "present_decision", "present_decision_with_caveats", "acquire_more_evidence"
    )
    assert "decision_brief" in result["consensus"]
    assert len(result["iterations"]) >= 1
    assert "acquisition_verifier" in result["verification"]
    assert "sequential_summary" in result
    assert result["sequential_summary"]["total_cost"] >= 0.0
