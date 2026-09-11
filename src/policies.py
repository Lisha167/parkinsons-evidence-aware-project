"""
policies.py
---------------------------------------------------------------
Acquisition Policy Benchmarks for Comparative Empirical Evaluation.

Implements 6 distinct acquisition selection policies (Section 24):
1. Random Assessment Policy
2. Fixed-Order Assessment Policy
3. Lowest-Cost Assessment Policy
4. Greedy Highest-Information Policy (Cost-blind)
5. Evidence-Gap Heuristic Policy
6. Proposed Cost/Burden-Aware Sequential Policy
"""
from typing import Dict, List, Any, Optional
import numpy as np

from src.schemas import AcquisitionDecision
from src.acquisition import (
    ASSESSMENT_CATALOG,
    recommend_next_assessment,
    estimate_expected_information_gain,
    compute_formal_utility,
)


class BaseAcquisitionPolicy:
    name: str = "base_policy"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        raise NotImplementedError


class RandomPolicy(BaseAcquisitionPolicy):
    name = "random_assessment"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        rng = rng or np.random.default_rng()
        available = [k for k in ASSESSMENT_CATALOG.keys() if k not in performed_assessments]
        if not available:
            available = list(ASSESSMENT_CATALOG.keys())
        chosen_id = str(rng.choice(available))
        defn = ASSESSMENT_CATALOG[chosen_id]
        return AcquisitionDecision(
            assessment=chosen_id,
            expected_info_gain=0.10,
            expected_loss_reduction=0.05,
            cost=defn.cost,
            burden=defn.burden,
            utility=-0.5 * (defn.cost + defn.burden),
            policy_name=self.name,
            description=defn.description,
            rationale=["Randomly chosen assessment benchmark."],
        )


class FixedOrderPolicy(BaseAcquisitionPolicy):
    name = "fixed_order"
    order = [
        "repeat_sustained_vowel",
        "reading_passage_task",
        "diadochokinetic_task",
        "accelerometer_gait_check",
        "clinical_updrs_exam",
    ]

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        for a_id in self.order:
            if a_id not in performed_assessments and a_id in ASSESSMENT_CATALOG:
                defn = ASSESSMENT_CATALOG[a_id]
                return AcquisitionDecision(
                    assessment=a_id,
                    expected_info_gain=0.15,
                    expected_loss_reduction=0.08,
                    cost=defn.cost,
                    burden=defn.burden,
                    utility=0.0,
                    policy_name=self.name,
                    description=defn.description,
                    rationale=[f"Fixed sequence step: {a_id}"],
                )
        fallback_id = "clinical_updrs_exam"
        defn = ASSESSMENT_CATALOG[fallback_id]
        return AcquisitionDecision(
            assessment=fallback_id,
            expected_info_gain=0.20,
            expected_loss_reduction=0.10,
            cost=defn.cost,
            burden=defn.burden,
            utility=0.0,
            policy_name=self.name,
            description=defn.description,
            rationale=["Fallback fixed order exam."],
        )


class LowestCostPolicy(BaseAcquisitionPolicy):
    name = "lowest_cost"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        available = [k for k in ASSESSMENT_CATALOG.keys() if k not in performed_assessments]
        if not available:
            available = list(ASSESSMENT_CATALOG.keys())
        # Sort purely by cost
        best_id = min(available, key=lambda k: ASSESSMENT_CATALOG[k].cost)
        defn = ASSESSMENT_CATALOG[best_id]
        return AcquisitionDecision(
            assessment=best_id,
            expected_info_gain=0.08,
            expected_loss_reduction=0.04,
            cost=defn.cost,
            burden=defn.burden,
            utility=-defn.cost,
            policy_name=self.name,
            description=defn.description,
            rationale=[f"Greedy lowest cost selection: {best_id}"],
        )


class HighestInformationPolicy(BaseAcquisitionPolicy):
    name = "highest_information_greedy"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        # Recommends with zero cost/burden penalties
        return recommend_next_assessment(
            evidence=evidence,
            pred_summary=pred_summary,
            lambda_cost=0.0,
            lambda_burden=0.0,
            excluded_assessments=performed_assessments,
        )


class EvidenceGapHeuristicPolicy(BaseAcquisitionPolicy):
    name = "evidence_gap_heuristic"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        # Simple heuristic gap matching with linear discount
        gaps = evidence.get("dominant_gaps", ["high_predictive_entropy"])
        available = [k for k in ASSESSMENT_CATALOG.keys() if k not in performed_assessments]
        if not available:
            available = list(ASSESSMENT_CATALOG.keys())
        
        best_id = available[0]
        best_score = -1.0
        for a_id in available:
            defn = ASSESSMENT_CATALOG[a_id]
            score = sum(defn.target_gaps.get(g, 0.0) for g in gaps) - 0.3 * (defn.cost + defn.burden)
            if score > best_score:
                best_score = score
                best_id = a_id
        
        defn = ASSESSMENT_CATALOG[best_id]
        return AcquisitionDecision(
            assessment=best_id,
            expected_info_gain=0.15,
            expected_loss_reduction=0.07,
            cost=defn.cost,
            burden=defn.burden,
            utility=round(best_score, 3),
            policy_name=self.name,
            description=defn.description,
            rationale=[f"Heuristic gap matching score: {best_score:.2f}"],
        )


class ProposedCostBurdenAwarePolicy(BaseAcquisitionPolicy):
    name = "proposed_cost_burden_aware"

    def select_next(
        self,
        evidence: Dict[str, Any],
        pred_summary: Dict[str, Any],
        performed_assessments: List[str],
        rng: Optional[np.random.Generator] = None,
    ) -> AcquisitionDecision:
        return recommend_next_assessment(
            evidence=evidence,
            pred_summary=pred_summary,
            lambda_cost=0.50,
            lambda_burden=0.30,
            excluded_assessments=performed_assessments,
        )


POLICY_REGISTRY: Dict[str, BaseAcquisitionPolicy] = {
    "random": RandomPolicy(),
    "fixed_order": FixedOrderPolicy(),
    "lowest_cost": LowestCostPolicy(),
    "highest_info": HighestInformationPolicy(),
    "gap_heuristic": EvidenceGapHeuristicPolicy(),
    "proposed": ProposedCostBurdenAwarePolicy(),
}
