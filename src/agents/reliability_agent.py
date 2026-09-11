"""
reliability_agent.py
---------------------------------------------------------------
Independent Predictive Reliability & Model Disagreement Verification Agent.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import VerificationResult


class ReliabilityAgent(BaseAgent):
    name = "reliability_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ev = context.get("evidence", {})
        pred = context.get("prediction", {})
        
        disagreement = float(pred.get("model_disagreement", ev.get("model_disagreement", 0.0)))
        margin = float(pred.get("confidence_margin", ev.get("confidence_margin", 0.5)))
        entropy = float(pred.get("predictive_entropy", ev.get("predictive_entropy", 0.5)))

        passed = disagreement <= 0.12 and margin >= 0.35 and entropy <= 0.88
        verdict = "acceptable" if passed else ("concern" if disagreement <= 0.18 else "rejected")

        supporting: List[str] = []
        concerns: List[str] = []

        if disagreement <= 0.12:
            supporting.append(f"Ensemble models exhibit high consensus (disagreement std={disagreement:.2f}).")
        else:
            concerns.append(f"Elevated ensemble model disagreement (std={disagreement:.2f} > 0.12).")

        if margin >= 0.35:
            supporting.append(f"Confidence margin ({margin:.2f}) indicates clear decision boundary separation.")
        else:
            concerns.append(f"Low confidence margin ({margin:.2f}) near boundary.")

        if entropy > 0.88:
            concerns.append(f"High predictive entropy ({entropy:.2f} bits).")

        fallback = (
            f"Predictive reliability judged {verdict.upper()}. Model disagreement std={disagreement:.2f}, "
            f"confidence margin={margin:.2f}, entropy={entropy:.2f} bits. "
            + (f"Concerns: {'; '.join(concerns)}" if concerns else "High model agreement verified.")
        )
        prompt = (
            "You are a clinical AI model reliability auditor for Parkinson's assessment. "
            f"Model disagreement={disagreement:.2f}, margin={margin:.2f}, entropy={entropy:.2f}. "
            f"Verdict: {verdict}. In 2 concise sentences, state whether model consensus is reliable."
        )
        explanation = self.narrate(prompt, fallback)

        res = VerificationResult(
            verifier_id=self.name,
            passed=passed,
            verdict=verdict,
            confidence=round(max(0.0, 1.0 - disagreement * 3.0), 2),
            supporting_evidence=supporting,
            concerns=concerns,
            contradictions=[] if passed else ["High epistemic model disagreement"],
            recommended_action="proceed" if passed else "acquire_syllable_or_clinical_confirmation",
            explanation=explanation,
            metrics={"model_disagreement": disagreement, "confidence_margin": margin, "entropy": entropy},
        )
        return res.to_dict()
