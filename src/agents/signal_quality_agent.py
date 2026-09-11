"""
signal_quality_agent.py
---------------------------------------------------------------
Independent Signal Quality Verification Agent.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import VerificationResult


class SignalQualityAgent(BaseAgent):
    name = "signal_quality_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ev = context.get("evidence", {})
        sq = float(ev.get("signal_quality", 1.0))
        passed = sq >= 0.55
        verdict = "acceptable" if passed else "rejected"

        supporting: List[str] = []
        concerns: List[str] = []
        if passed:
            supporting.append(f"Recording signal quality ({sq:.2f}) exceeds minimum acceptability threshold (0.55).")
            if sq >= 0.80:
                supporting.append("High acoustic fidelity with minimal noise artifacts.")
        else:
            concerns.append(f"Recording signal quality ({sq:.2f}) is below 0.55 acceptability threshold.")

        fallback = (
            f"Recording signal quality score is {sq:.2f}. "
            f"Judged {verdict.upper()} for downstream clinical analysis."
        )
        prompt = (
            "You are a clinical audio-quality auditor for Parkinson's voice assessment. "
            f"Signal quality is {sq:.2f} (threshold 0.55). Judged {verdict}. "
            "In 2 concise sentences, summarize recording suitability and caveats for a clinician."
        )
        explanation = self.narrate(prompt, fallback)

        res = VerificationResult(
            verifier_id=self.name,
            passed=passed,
            verdict=verdict,
            confidence=round(sq, 2),
            supporting_evidence=supporting,
            concerns=concerns,
            contradictions=[] if passed else ["Sub-threshold audio fidelity"],
            recommended_action="proceed" if passed else "repeat_acoustic_recording",
            explanation=explanation,
            metrics={"signal_quality": sq},
        )
        return res.to_dict()
