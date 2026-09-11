"""
adversarial_agent.py
---------------------------------------------------------------
Adversarial Verification Critic.

Systematically challenges diagnostic confidence, evidence sufficiency,
and acquisition economy to prevent overconfident or unsubstantiated clinical outputs.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import AdversarialChallenge


class AdversarialAgent(BaseAgent):
    name = "adversarial_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        verdicts = context.get("agent_verdicts", [])
        ev = context.get("evidence", {})
        pred = context.get("prediction", {})
        acq = context.get("acquisition", {})
        rec = acq.get("recommendation", acq) if isinstance(acq, dict) else {}

        challenges: List[str] = []
        challenge_targets: List[str] = []

        # Find specific verifiers
        verdict_map = {v["verifier_id"]: v for v in verdicts if "verifier_id" in v}
        if not verdict_map:
            verdict_map = {v.get("agent", ""): v for v in verdicts}

        rel = verdict_map.get("reliability_agent", {})
        sig = verdict_map.get("signal_quality_agent", {})
        phen = verdict_map.get("phenotype_agent", {})
        temp = verdict_map.get("temporal_agent", {})
        acq_v = verdict_map.get("acquisition_verifier", {})

        # 1. Challenge: Overconfidence with borderline model disagreement
        disagreement = float(pred.get("model_disagreement", 0.0))
        if disagreement > 0.08:
            challenges.append(
                f"Model disagreement (std={disagreement:.2f}) is borderline — caution against treating prediction as fully definitive."
            )
            challenge_targets.append("model_reliability")

        # 2. Challenge: Phenotype and prediction discordance
        phen_consistency = float(ev.get("phenotype_consistency", 1.0))
        if phen_consistency < 0.70:
            challenges.append(
                f"Phenotype-to-model consistency is weak ({phen_consistency:.2f}) — rule-based acoustic severity diverges from ML probability."
            )
            challenge_targets.append("phenotype_consistency")

        # 3. Challenge: Decision state claimed 'sufficient' with marginal signal quality
        sq = float(ev.get("signal_quality", 1.0))
        if ev.get("decision_state") == "sufficient" and sq < 0.65:
            challenges.append(
                f"Evidence is marked 'sufficient' but signal quality ({sq:.2f}) is only marginally above the 0.55 cutoff."
            )
            challenge_targets.append("signal_quality")

        # 4. Challenge: High PD probability without longitudinal baseline
        change_rep = temp.get("change_report", {}) if isinstance(temp, dict) else {}
        if not change_rep.get("has_baseline", True) and float(pred.get("pd_probability", 0.0)) > 0.68:
            challenges.append(
                "High disease probability is presented solely on cross-sectional data with no patient baseline for validation."
            )
            challenge_targets.append("temporal_baseline")

        # 5. Challenge: Unnecessary high-cost assessment when evidence is already sufficient
        if ev.get("decision_state") == "sufficient" and rec and float(rec.get("cost", 0.0)) > 0.20:
            challenges.append(
                f"Expensive follow-up ('{rec.get('assessment')}', cost={rec.get('cost')}) proposed despite sufficient baseline evidence."
            )
            challenge_targets.append("acquisition_necessity")

        # 6. Challenge: Acquisition verifier raised concerns
        if acq_v and not acq_v.get("passed", True):
            challenges.append(
                f"Independent acquisition verifier rejected/questioned the proposed test: {'; '.join(acq_v.get('concerns', []))}."
            )
            challenge_targets.append("acquisition_verifier_concerns")

        # 7. Challenge: Premature stopping under high entropy
        entropy = float(pred.get("predictive_entropy", 0.0))
        if ev.get("decision_state") == "sufficient" and entropy > 0.80:
            challenges.append(
                f"Decision state marked 'sufficient' despite elevated Shannon entropy ({entropy:.2f} bits)."
            )
            challenge_targets.append("premature_sufficiency")

        unsupported = len(challenges) > 0
        severity = "high" if len(challenges) >= 3 else ("medium" if len(challenges) >= 1 else "low")

        fallback = (
            f"Adversarial review raised {len(challenges)} challenge(s) [Severity: {severity.upper()}]: "
            + (" | ".join(challenges) if challenges else "Reasoning passed adversarial audit with no critical flaws.")
        )
        prompt = (
            "You are a rigorous adversarial clinical AI critic reviewing a Parkinson's decision pipeline. "
            f"Identified challenges: {challenges if challenges else 'none'}. "
            "In 2-3 concise sentences, explain to a neurologist why these vulnerabilities must be considered before action."
        )
        explanation = self.narrate(prompt, fallback)

        res = AdversarialChallenge(
            challenges=challenges,
            unsupported_reasoning_found=unsupported,
            severity=severity,
            challenge_targets=challenge_targets,
            explanation=explanation,
        )
        return res.to_dict()
