"""
phenotype_agent.py
---------------------------------------------------------------
Speech Phenotype Interpretation & Consistency Verification Agent.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import VerificationResult


class PhenotypeAgent(BaseAgent):
    name = "phenotype_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ev = context.get("evidence", {})
        phen = context.get("phenotype", {})
        pred = context.get("prediction", {})
        
        consistency = float(ev.get("phenotype_consistency", 0.8))
        sev = float(phen.get("phenotype_overall_phenotype_severity",
                    phen.get("overall_phenotype_severity", 0.5)))
        p_prob = float(pred.get("pd_probability", 0.5))

        passed = consistency >= 0.60
        verdict = "acceptable" if passed else "concern"

        supporting: List[str] = []
        concerns: List[str] = []

        if passed:
            supporting.append(f"Acoustic phenotype severity ({sev:.2f}) is consistent with model probability ({p_prob:.2f}).")
        else:
            concerns.append(f"Discordance between rule-based phenotype ({sev:.2f}) and model probability ({p_prob:.2f}).")

        dominant_axis = "vocal_stability"
        if isinstance(phen, dict):
            axes = {k.replace("phenotype_", ""): v for k, v in phen.items() if "vocal" in k or "amplitude" in k or "noise" in k or "dynamics" in k}
            if axes:
                dominant_axis = max(axes, key=lambda k: axes[k])

        fallback = (
            f"Phenotype interpretation judged {verdict.upper()}. Consistency score is {consistency:.2f}. "
            f"Primary phenotypic severity is {sev:.2f} driven by '{dominant_axis}'."
        )
        prompt = (
            "You are a speech-language pathologist AI auditor reviewing acoustic phenotype findings for Parkinson's disease. "
            f"Phenotypic severity is {sev:.2f} (dominant: {dominant_axis}), model PD prob is {p_prob:.2f}, consistency is {consistency:.2f}. "
            "In 2 sentences, summarize clinical phenotypic findings and acoustic consistency. No bullet points."
        )
        explanation = self.narrate(prompt, fallback)

        res = VerificationResult(
            verifier_id=self.name,
            passed=passed,
            verdict=verdict,
            confidence=round(consistency, 2),
            supporting_evidence=supporting,
            concerns=concerns,
            contradictions=[] if passed else ["Cross-axis phenotypic discordance"],
            recommended_action="proceed" if passed else "acquire_reading_passage_evaluation",
            explanation=explanation,
            metrics={"phenotype_consistency": consistency, "overall_severity": sev, "dominant_axis": dominant_axis},
        )
        return res.to_dict()
