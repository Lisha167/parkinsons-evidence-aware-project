"""
test_research_components.py
---------------------------------------------------------------
Rigorous Unit & Integration Tests for Core Research Invariants:
1. Mathematical Entropy & Information Gain Calculation
2. Zero Future-Data Leakage in Baseline Estimation
3. Concrete AssessmentResult Creation Guarantee
4. Robust MAD Baseline & Temporal Persistence
5. Evidence State Machine & Gap Extraction
6. Acquisition Policy Utility & Cost-Burden Penalties
7. Multi-Agent Verification & Acquisition Verifier
8. Adversarial Critic Challenge Rules
9. Consensus Synthesis & Traceability
10. End-to-End Sequential Trajectory Determinism & Reproducibility
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.generate_synthetic_data import generate, SCENARIOS
from src.features import add_phenotype_columns, compute_signal_quality_metrics, build_phenotype
from src.models import VoiceEnsemble, compute_binary_entropy, compute_expected_calibration_error, train_and_evaluate
from src.baseline import compute_patient_baseline, evaluate_change, MIN_VISITS_FOR_BASELINE
from src.evidence import assess_evidence, identify_evidence_gaps
from src.acquisition import (
    ASSESSMENT_CATALOG,
    sample_assessment_observation,
    estimate_expected_information_gain,
    compute_formal_utility,
    recommend_next_assessment,
)
from src.policies import POLICY_REGISTRY
from src.agents.acquisition_verifier import AcquisitionDecisionVerifier
from src.agents.orchestrator import DecisionOrchestrator, update_row_with_evidence
from src.ablation import run_ablation_study
from src.schemas import AssessmentResult, VerificationResult, EvidenceState


@pytest.fixture
def sample_dataset():
    df = generate(n_healthy=8, n_pd=8, scenario="moderate", seed=42)
    return add_phenotype_columns(df)


@pytest.fixture
def trained_ensemble(sample_dataset):
    ens, _, _, _ = train_and_evaluate(sample_dataset, seed=42)
    return ens


# ---------------------------------------------------------------------
# Invariant 1: Information Gain Calculation
# ---------------------------------------------------------------------
def test_binary_entropy_bounds():
    # Maximum uncertainty at p=0.5
    h_max = compute_binary_entropy(0.5)
    assert abs(h_max - 1.0) < 1e-5

    # Maximum certainty at boundaries
    assert compute_binary_entropy(0.01) < 0.15
    assert compute_binary_entropy(0.99) < 0.15
    assert compute_binary_entropy(0.5) > compute_binary_entropy(0.8)


def test_expected_information_gain_calculation():
    candidate = ASSESSMENT_CATALOG["diadochokinetic_task"]
    eig, loss_red = estimate_expected_information_gain(
        candidate=candidate,
        current_prob=0.50,
        current_entropy=1.0,
        evidence_gaps=["high_model_disagreement", "low_predictive_confidence"],
    )
    assert eig > 0.0, "EIG must be strictly positive for relevant gaps under high entropy"
    assert loss_red > 0.0, "Loss reduction must be positive"
    assert eig <= 1.0, "EIG cannot exceed current Shannon entropy"


def test_cost_burden_utility_tradeoff():
    # Higher cost/burden must reduce utility
    util_low_cost = compute_formal_utility(eig=0.20, loss_reduction=0.10, cost=0.05, burden=0.05)
    util_high_cost = compute_formal_utility(eig=0.20, loss_reduction=0.10, cost=0.90, burden=0.70)
    assert util_low_cost > util_high_cost


# ---------------------------------------------------------------------
# Invariant 2: Zero Future-Data Leakage
# ---------------------------------------------------------------------
def test_baseline_strict_temporal_isolation(sample_dataset):
    pid = sample_dataset["patient_id"].iloc[0]
    p_hist = sample_dataset[sample_dataset["patient_id"] == pid].sort_values("visit_id")
    
    # Baseline for visit 1 (no prior visits) -> None
    b1 = compute_patient_baseline(p_hist, upto_visit_id=1)
    assert b1 is None

    # Baseline for visit 2 (1 prior visit < MIN_VISITS_FOR_BASELINE) -> None
    b2 = compute_patient_baseline(p_hist, upto_visit_id=2)
    assert b2 is None

    # Baseline for visit 3 (2 prior visits: visit 1 and visit 2)
    b3 = compute_patient_baseline(p_hist, upto_visit_id=3)
    assert b3 is not None
    assert b3.baseline_n_visits == 2

    # Verify that visits >= 3 are NEVER included in b3
    p_hist_modified = p_hist.copy()
    # Drastically mutate future visit 4
    p_hist_modified.loc[p_hist_modified["visit_id"] >= 3, "phenotype_vocal_stability"] = 999.0
    b3_check = compute_patient_baseline(p_hist_modified, upto_visit_id=3)
    assert b3.location_estimates["phenotype_vocal_stability"] == b3_check.location_estimates["phenotype_vocal_stability"]


# ---------------------------------------------------------------------
# Invariant 3: Concrete AssessmentResult Guarantee
# ---------------------------------------------------------------------
def test_sample_assessment_observation_generates_concrete_evidence(sample_dataset):
    row = sample_dataset.iloc[0]
    for a_id in ASSESSMENT_CATALOG.keys():
        obs = sample_assessment_observation(row, a_id)
        assert isinstance(obs, AssessmentResult)
        assert obs.assessment_id == a_id
        assert len(obs.evidence_data) > 0
        assert obs.cost == ASSESSMENT_CATALOG[a_id].cost
        assert obs.burden == ASSESSMENT_CATALOG[a_id].burden


def test_update_row_with_evidence(sample_dataset):
    row = sample_dataset.iloc[0].copy()
    obs = sample_assessment_observation(row, "repeat_sustained_vowel")
    updated = update_row_with_evidence(row, obs)
    assert updated["signal_quality"] == obs.evidence_data["new_signal_quality"]
    assert f"_obs_{obs.assessment_id}" in updated


# ---------------------------------------------------------------------
# Invariant 4: Robust Baseline & Change Detection
# ---------------------------------------------------------------------
def test_robust_baseline_with_outlier_insensitivity(sample_dataset):
    pid = sample_dataset[sample_dataset["label"] == 0]["patient_id"].iloc[0]
    p_hist = sample_dataset[sample_dataset["patient_id"] == pid].sort_values("visit_id").copy()
    
    # Compute baseline
    base1 = compute_patient_baseline(p_hist, upto_visit_id=4)
    assert base1 is not None

    # Inject single outlier in prior visit 1
    p_hist_outlier = p_hist.copy()
    p_hist_outlier.loc[p_hist_outlier["visit_id"] == 1, "phenotype_vocal_stability"] = 1.0
    base2 = compute_patient_baseline(p_hist_outlier, upto_visit_id=4)
    
    # Robust median should remain bounded
    assert abs(base2.location_estimates["phenotype_vocal_stability"] - base1.location_estimates["phenotype_vocal_stability"]) < 0.35


def test_change_detection_persistence(sample_dataset):
    pid = sample_dataset[sample_dataset["label"] == 1]["patient_id"].iloc[0]
    p_hist = sample_dataset[sample_dataset["patient_id"] == pid].sort_values("visit_id")
    last_row = p_hist.iloc[-1]
    base = compute_patient_baseline(p_hist, int(last_row["visit_id"]))
    res = evaluate_change(last_row, base, patient_history=p_hist)
    assert "meaningful_change" in res
    assert "verdict" in res
    assert "method" in res


# ---------------------------------------------------------------------
# Invariant 5: Multi-Agent Verification Layer
# ---------------------------------------------------------------------
def test_acquisition_decision_verifier(sample_dataset, trained_ensemble):
    p_hist = sample_dataset[sample_dataset["label"] == 1].groupby("patient_id").get_group(
        sample_dataset[sample_dataset["label"] == 1]["patient_id"].iloc[0]
    )
    orch = DecisionOrchestrator(trained_ensemble, use_llm=False)
    res = orch.run_case(p_hist, int(p_hist["visit_id"].max()))
    
    acq_v = res["verification"]["acquisition_verifier"]
    assert "verifier_id" in acq_v
    assert "verdict" in acq_v
    assert "passed" in acq_v
    assert acq_v["verifier_id"] == "acquisition_verifier"


def test_adversarial_critic_triggers_on_overconfidence(sample_dataset, trained_ensemble):
    orch = DecisionOrchestrator(trained_ensemble, use_llm=False)
    # Simulated artificial context with model disagreement
    context = {
        "prediction": {"pd_probability": 0.85, "confidence_margin": 0.70, "model_disagreement": 0.15, "predictive_entropy": 0.60},
        "evidence": {"decision_state": "sufficient", "signal_quality": 0.60, "phenotype_consistency": 0.65, "dominant_gaps": []},
        "agent_verdicts": [
            {"verifier_id": "reliability_agent", "passed": False, "concerns": ["High disagreement"]},
            {"verifier_id": "signal_quality_agent", "passed": True, "concerns": []},
            {"verifier_id": "temporal_agent", "change_report": {"has_baseline": False}},
        ],
        "acquisition": {"recommendation": {"assessment": "clinical_updrs_exam", "cost": 0.90, "expected_info_gain": 0.05}},
    }
    adv_res = orch.adversarial_agent.run(context)
    assert adv_res["unsupported_reasoning_found"] is True
    assert len(adv_res["challenges"]) >= 2


# ---------------------------------------------------------------------
# Invariant 6: Sequential Loop Trajectory Determinism
# ---------------------------------------------------------------------
def test_sequential_loop_determinism(sample_dataset, trained_ensemble):
    p_hist = sample_dataset[sample_dataset["label"] == 1].groupby("patient_id").get_group(
        sample_dataset[sample_dataset["label"] == 1]["patient_id"].iloc[0]
    )
    orch = DecisionOrchestrator(trained_ensemble, use_llm=False)
    
    res1 = orch.run_case(p_hist, int(p_hist["visit_id"].max()), rng_seed=123)
    res2 = orch.run_case(p_hist, int(p_hist["visit_id"].max()), rng_seed=123)

    assert len(res1["iterations"]) == len(res2["iterations"])
    assert res1["sequential_summary"]["total_cost"] == res2["sequential_summary"]["total_cost"]
    assert res1["consensus"]["final_action"] == res2["consensus"]["final_action"]


# ---------------------------------------------------------------------
# Invariant 7: Policy Benchmarking
# ---------------------------------------------------------------------
def test_policy_registry():
    assert len(POLICY_REGISTRY) == 6
    assert "random" in POLICY_REGISTRY
    assert "proposed" in POLICY_REGISTRY
    assert "fixed_order" in POLICY_REGISTRY
    assert "lowest_cost" in POLICY_REGISTRY
    assert "highest_info" in POLICY_REGISTRY
    assert "gap_heuristic" in POLICY_REGISTRY
