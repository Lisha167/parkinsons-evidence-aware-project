"""
acquisition_agent.py
---------------------------------------------------------------
Information-Acquisition Recommendation Agent.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.acquisition import recommend_next_assessment


class AcquisitionAgent(BaseAgent):
    name = "acquisition_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ev = context.get("evidence", {})
        pred = context.get("prediction", {})
        performed = context.get("performed_assessments", [])

        rec = recommend_next_assessment(
            evidence=ev,
            pred_summary=pred,
            excluded_assessments=performed,
        )

        fallback = (
            f"Recommended next assessment: '{rec.assessment}' ({rec.description}). "
            f"Expected info gain = {rec.expected_info_gain:.3f} bits, cost = {rec.cost:.2f}, "
            f"burden = {rec.burden:.2f}, utility = {rec.utility:.3f}. Rationale: {'; '.join(rec.rationale)}"
        )
        prompt = (
            "You are an information-acquisition planning agent for Parkinson's decision support. "
            f"Current state: {ev.get('decision_state')}, gaps: {ev.get('dominant_gaps')}. "
            f"Recommended assessment: '{rec.assessment}' (EIG={rec.expected_info_gain:.3f} bits, cost={rec.cost:.2f}, burden={rec.burden:.2f}). "
            "In 2-3 concise sentences, justify this selection and cost/burden trade-off to a clinician."
        )
        explanation = self.narrate(prompt, fallback)

        return {
            "agent": self.name,
            "recommendation": rec.to_dict(),
            "explanation": explanation,
        }
