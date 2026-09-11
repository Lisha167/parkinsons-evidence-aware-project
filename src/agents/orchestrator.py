"""
orchestrator.py
---------------------------------------------------------------
Sequential Decision-Support Orchestrator.

Implements the complete evidence-aware sequential decision loop:
1. Acoustic Screening + Phenotype Extraction
2. Patient-Specific Baseline & Change Validation
3. Computational Evidence Assessment
4. Formal Next-Best Assessment Selection (EIG - Cost - Burden)
5. Probabilistic Evidence Observation & Belief Update Cycle
6. Multi-Agent Verification (5 Independent Verifiers + Adversarial Critic + Consensus)
7. Traceable Clinician-Facing Decision Trajectory
"""
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.features import build_phenotype, compute_signal_quality_metrics
from src.evidence import assess_evidence
from src.baseline import compute_patient_baseline, evaluate_change
from src.acquisition import (
    sample_assessment_observation,
    recommend_next_assessment,
    ASSESSMENT_CATALOG,
)
from src.agents.signal_quality_agent import SignalQualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.phenotype_agent import PhenotypeAgent
from src.agents.temporal_agent import TemporalAgent
from src.agents.acquisition_verifier import AcquisitionDecisionVerifier
from src.agents.acquisition_agent import AcquisitionAgent
from src.agents.adversarial_agent import AdversarialAgent
from src.agents.consensus_agent import ConsensusAgent
from src.models import compute_binary_entropy
from src.schemas import DecisionIteration, DecisionState, AssessmentResult

MAX_REASSESSMENT_ITERATIONS = 4
MAX_ASSESSMENT_BUDGET_COST = 1.80


def update_row_with_evidence(current_row: pd.Series, obs_result: AssessmentResult) -> pd.Series:
    """Updates the feature row and evidentiary state with real observation data."""
    new_row = current_row.copy()
    data = obs_result.evidence_data

    # Update acoustic or signal attributes
    if "new_signal_quality" in data:
        new_row["signal_quality"] = data["new_signal_quality"]
    if "jitter_pct" in data:
        new_row["jitter_pct"] = data["jitter_pct"]
    if "shimmer" in data:
        new_row["shimmer"] = data["shimmer"]
    if "spread1" in data:
        new_row["spread1"] = data["spread1"]
    if "rpde" in data:
        new_row["rpde"] = data["rpde"]
    if "ppe" in data:
        new_row["ppe"] = data["ppe"]

    # Store observation object in row metadata
    new_row[f"_obs_{obs_result.assessment_id}"] = True
    return new_row


class DecisionOrchestrator:
    def __init__(self, ensemble, use_llm: bool = False, llm_model: str = "llama3.2:1b"):
        self.ensemble = ensemble
        agent_kwargs = dict(use_llm=use_llm, model=llm_model)
        self.signal_agent = SignalQualityAgent(**agent_kwargs)
        self.reliability_agent = ReliabilityAgent(**agent_kwargs)
        self.phenotype_agent = PhenotypeAgent(**agent_kwargs)
        self.temporal_agent = TemporalAgent(**agent_kwargs)
        self.acq_verifier = AcquisitionDecisionVerifier(**agent_kwargs)
        self.acquisition_agent = AcquisitionAgent(**agent_kwargs)
        self.adversarial_agent = AdversarialAgent(**agent_kwargs)
        self.consensus_agent = ConsensusAgent(**agent_kwargs)

    def _score_row(self, row: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        phen = build_phenotype(row)
        phen_dict = {f"phenotype_{k}": v for k, v in phen.items()}
        sq_metrics = compute_signal_quality_metrics(row)
        phen_dict["signal_quality"] = sq_metrics.overall_quality

        row_df = pd.DataFrame([row])
        for k, v in phen_dict.items():
            row_df[k] = v

        pred_summary = self.ensemble.predict_summary(row_df).iloc[0].to_dict()
        return pred_summary, phen_dict, sq_metrics

    def run_case(
        self,
        patient_history: pd.DataFrame,
        visit_id: int,
        rng_seed: int = 42,
        max_iterations: int = MAX_REASSESSMENT_ITERATIONS,
        cost_budget: float = MAX_ASSESSMENT_BUDGET_COST,
    ) -> Dict[str, Any]:
        """Runs the complete sequential decision process for a single patient visit."""
        rng = np.random.default_rng(rng_seed)
        patient_records = patient_history.sort_values("visit_id").reset_index(drop=True)
        current_rows = patient_records[patient_records["visit_id"] == visit_id]
        if len(current_rows) == 0:
            raise ValueError(f"Visit ID {visit_id} not found for patient.")
        
        current_row = current_rows.iloc[0].copy()
        patient_id = str(current_row["patient_id"])

        # Compute baseline strictly from prior visits (zero leakage)
        baseline = compute_patient_baseline(patient_records, visit_id)

        iterations: List[Dict[str, Any]] = []
        performed_assessments: List[str] = []
        accumulated_observations: List[Dict[str, Any]] = []
        cumulative_cost = 0.0
        cumulative_burden = 0.0

        for it in range(max_iterations):
            pred_summary, phen, sq_metrics = self._score_row(current_row)
            change_report = evaluate_change(
                pd.Series({**phen, **{c: current_row[c] for c in current_row.index if c not in phen}}),
                baseline,
                patient_history=patient_records,
            )
            evidence_report = assess_evidence(pred_summary, phen, change_report, signal_metrics=sq_metrics)
            evidence_dict = evidence_report.to_dict()

            entropy_current = float(pred_summary["predictive_entropy"])
            entropy_prev = iterations[-1]["prediction"]["predictive_entropy"] if it > 0 else entropy_current
            uncertainty_reduction = float(max(0.0, entropy_prev - entropy_current))

            iter_data: Dict[str, Any] = {
                "iteration": it,
                "prediction": pred_summary,
                "phenotype": phen,
                "signal_metrics": sq_metrics.to_dict(),
                "change_report": change_report,
                "evidence": evidence_dict,
                "entropy_before": entropy_prev,
                "entropy_after": entropy_current,
                "uncertainty_reduction": uncertainty_reduction,
                "cumulative_cost": cumulative_cost,
                "cumulative_burden": cumulative_burden,
            }

            # Stop condition 1: Evidence is sufficient
            if evidence_dict["decision_state"] == "sufficient":
                iterations.append(iter_data)
                break

            # Stop condition 2: Budget exhausted
            if cumulative_cost >= cost_budget:
                iter_data["budget_exhausted"] = True
                iterations.append(iter_data)
                break

            # Select next-best assessment
            acq_context = {
                "evidence": evidence_dict,
                "prediction": pred_summary,
                "phenotype": phen,
                "performed_assessments": performed_assessments,
            }
            acq_result = self.acquisition_agent.run(acq_context)
            rec = acq_result["recommendation"]
            chosen_assessment = rec["assessment"]

            iter_data["acquisition"] = acq_result

            # Stop condition 3: Max iterations reached
            if it == max_iterations - 1:
                iterations.append(iter_data)
                break

            # Generate real probabilistic assessment observation
            obs_result = sample_assessment_observation(current_row, chosen_assessment, rng=rng)
            performed_assessments.append(chosen_assessment)
            accumulated_observations.append(obs_result.to_dict())
            cumulative_cost += obs_result.cost
            cumulative_burden += obs_result.burden

            iter_data["assessment_result"] = obs_result.to_dict()
            iterations.append(iter_data)

            # Incorporate new evidence and update state for next iteration
            current_row = update_row_with_evidence(current_row, obs_result)

        final_iter = iterations[-1]
        final_context = {
            "evidence": final_iter["evidence"],
            "prediction": final_iter["prediction"],
            "phenotype": final_iter["phenotype"],
            "change_report": final_iter["change_report"],
            "performed_assessments": performed_assessments,
        }

        # Multi-Agent Verification Layer (Section 16, 17)
        sig_v = self.signal_agent.run(final_context)
        rel_v = self.reliability_agent.run(final_context)
        phen_v = self.phenotype_agent.run(final_context)
        temp_v = self.temporal_agent.run(final_context)

        # Acquisition Decision Verifier
        if "acquisition" in final_iter:
            acq_v = self.acq_verifier.run({**final_context, "acquisition": final_iter["acquisition"]})
        else:
            final_iter["acquisition"] = self.acquisition_agent.run(final_context)
            acq_v = self.acq_verifier.run({**final_context, "acquisition": final_iter["acquisition"]})

        agent_verdicts = [sig_v, rel_v, phen_v, temp_v, acq_v]

        # Adversarial Critic (Section 18)
        adv_context = {
            **final_context,
            "agent_verdicts": agent_verdicts,
            "acquisition": final_iter["acquisition"],
        }
        adv_v = self.adversarial_agent.run(adv_context)

        # Consensus Engine (Section 19, 20)
        consensus_context = {
            **final_context,
            "agent_verdicts": agent_verdicts,
            "adversarial": adv_v,
            "acquisition": final_iter["acquisition"],
        }
        consensus_v = self.consensus_agent.run(consensus_context)

        # Compute sequential summary metrics (Section 21, 22)
        initial_entropy = iterations[0]["prediction"]["predictive_entropy"]
        final_entropy = final_iter["prediction"]["predictive_entropy"]
        total_entropy_reduction = float(max(0.0, initial_entropy - final_entropy))
        rel_uncertainty_reduction = total_entropy_reduction / initial_entropy if initial_entropy > 0 else 0.0

        sequential_summary = {
            "total_assessments": len(performed_assessments),
            "assessments_performed": performed_assessments,
            "total_cost": round(cumulative_cost, 3),
            "total_burden": round(cumulative_burden, 3),
            "iterations_count": len(iterations),
            "initial_entropy": round(initial_entropy, 3),
            "final_entropy": round(final_entropy, 3),
            "absolute_uncertainty_reduction": round(total_entropy_reduction, 3),
            "relative_uncertainty_reduction": round(rel_uncertainty_reduction, 3),
            "uncertainty_reduction_per_cost": round(total_entropy_reduction / max(0.01, cumulative_cost), 3),
            "resolved_to_sufficient": final_iter["evidence"]["decision_state"] == "sufficient",
        }

        return {
            "patient_id": patient_id,
            "visit_id": visit_id,
            "ground_truth_label": int(current_row.get("label", 0)),
            "iterations": iterations,
            "final_evidence": final_iter["evidence"],
            "verification": {
                "signal_quality": sig_v,
                "reliability": rel_v,
                "phenotype": phen_v,
                "temporal": temp_v,
                "acquisition_verifier": acq_v,
            },
            "adversarial": adv_v,
            "acquisition": final_iter["acquisition"],
            "consensus": consensus_v,
            "sequential_summary": sequential_summary,
            "accumulated_observations": accumulated_observations,
        }
