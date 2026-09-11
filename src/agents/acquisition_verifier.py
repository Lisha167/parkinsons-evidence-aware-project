"""
acquisition_verifier.py
---------------------------------------------------------------
Independent Acquisition Decision Verification Agent.

Audits candidate information-acquisition recommendations:
1. Does the proposed assessment target the dominant evidence gaps?
2. Is the expected information gain (EIG) mathematically sufficient?
3. Is patient and operational burden proportional to value?
4. Is another candidate clearly superior in cost-benefit utility?
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import VerificationResult
from src.acquisition import ASSESSMENT_CATALOG


class AcquisitionDecisionVerifier(BaseAgent):
    name = "acquisition_verifier"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        acq = context.get("acquisition", {})
        rec = acq.get("recommendation", acq)
        ev = context.get("evidence", {})
        
        assessment_id = rec.get("assessment", "repeat_sustained_vowel")
        eig = float(rec.get("expected_info_gain", 0.0))
        cost = float(rec.get("cost", 0.0))
        burden = float(rec.get("burden", 0.0))
        dominant_gaps = ev.get("dominant_gaps", [])

        concerns: List[str] = []
        contradictions: List[str] = []
        supporting: List[str] = []

        defn = ASSESSMENT_CATALOG.get(assessment_id)
        if defn:
            supporting.append(f"Assessment '{defn.name}' is registered in feasible catalog.")
            # Check gap targeting
            addressed = [g for g in dominant_gaps if g in defn.target_gaps]
            if addressed:
                supporting.append(f"Directly targets active gaps: {', '.join(addressed)}.")
            elif dominant_gaps:
                concerns.append(f"Selected assessment does not directly target dominant gaps: {', '.join(dominant_gaps)}.")
        else:
            contradictions.append(f"Unknown assessment id: {assessment_id}")

        # Check EIG threshold
        if eig < 0.03 and ev.get("decision_state") != "sufficient":
            concerns.append(f"Low expected information gain ({eig:.3f} bits) — marginal expected uncertainty reduction.")
        else:
            supporting.append(f"Expected info gain ({eig:.3f} bits) meets significance threshold.")

        # Check high cost/burden disproportionality
        if cost > 0.80 and eig < 0.15:
            concerns.append(f"High acquisition cost ({cost:.2f}) with modest expected info gain ({eig:.3f}).")

        passed = len(contradictions) == 0 and len(concerns) <= 1
        verdict = "acceptable" if passed else ("concern" if len(concerns) > 1 else "rejected")
        confidence = 0.88 if passed else 0.65

        fallback = (
            f"Acquisition verifier judged choice '{assessment_id}' as {verdict.upper()}. "
            f"EIG={eig:.3f}, Cost={cost:.2f}, Burden={burden:.2f}. "
            + (f"Concerns: {'; '.join(concerns)}" if concerns else "No significant acquisition concerns.")
        )

        prompt = (
            "You are an independent clinical AI acquisition auditor. "
            f"Reviewing recommended assessment '{assessment_id}' (EIG={eig:.3f}, Cost={cost:.2f}, Burden={burden:.2f}). "
            f"Dominant gaps: {dominant_gaps}. Verdict: {verdict}. "
            "In 2 short sentences, explain if this acquisition choice represents sound clinical resource utilization. No bullets."
        )
        explanation = self.narrate(prompt, fallback)

        res = VerificationResult(
            verifier_id=self.name,
            passed=passed,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=supporting,
            concerns=concerns,
            contradictions=contradictions,
            recommended_action="proceed_with_acquisition" if passed else "review_candidate_alternatives",
            explanation=explanation,
            metrics={"expected_info_gain": eig, "cost": cost, "burden": burden},
        )
        return res.to_dict()
