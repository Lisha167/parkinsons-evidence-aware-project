"""
temporal_agent.py
---------------------------------------------------------------
Longitudinal Change & Patient Baseline Verification Agent.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import VerificationResult


class TemporalAgent(BaseAgent):
    name = "temporal_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        change = context.get("change_report", {})
        ev = context.get("evidence", {})
        verdict_str = change.get("verdict", "no_baseline_insufficient_history")
        has_baseline = bool(change.get("has_baseline", False))
        is_meaningful = bool(change.get("meaningful_change", False))
        margin = float(ev.get("confidence_margin", 0.5))

        passed = not is_meaningful or margin >= 0.40
        verdict = "acceptable" if passed else "concern"

        supporting: List[str] = []
        concerns: List[str] = []

        if not has_baseline:
            concerns.append("No individualized baseline available yet (< 2 prior visits); population norms only.")
        else:
            n_v = change.get("baseline_n_visits", 0)
            max_dev = change.get("max_abs_deviation") or change.get("max_abs_z", 0.0)
            axis = change.get("driving_axis", "speech_features")
            supporting.append(f"Baseline established over {n_v} historical visits using robust median/MAD.")
            if is_meaningful:
                supporting.append(f"Sustained patient-specific change detected on '{axis}' (z_mad={max_dev:.2f}).")
            else:
                supporting.append("Observation is within expected patient-specific fluctuation range.")

        if is_meaningful and margin < 0.40:
            concerns.append("Longitudinal change detected but cross-sectional prediction has low confidence.")

        fallback = (
            f"Longitudinal change analysis: {verdict_str.replace('_', ' ')}. "
            + (f"Baseline visits: {change.get('baseline_n_visits', 0)}. " if has_baseline else "")
            + (f"Concerns: {'; '.join(concerns)}" if concerns else "Temporal stability verified.")
        )
        prompt = (
            "You are a clinical neuro-monitoring specialist reviewing longitudinal voice change in Parkinson's. "
            f"Verdict: {verdict_str}, has_baseline={has_baseline}. "
            "In 2 sentences, explain if this represents persistent progressive change or normal intra-individual fluctuation."
        )
        explanation = self.narrate(prompt, fallback)

        res = VerificationResult(
            verifier_id=self.name,
            passed=passed,
            verdict=verdict,
            confidence=0.85 if has_baseline else 0.50,
            supporting_evidence=supporting,
            concerns=concerns,
            contradictions=["Temporal-predictive tension"] if (is_meaningful and margin < 0.35) else [],
            recommended_action="proceed" if passed else "acquire_extended_home_monitoring",
            explanation=explanation,
            metrics={"has_baseline": has_baseline, "meaningful_change": is_meaningful, "change_report": change},
        )
        return res.to_dict()
