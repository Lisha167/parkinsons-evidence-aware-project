"""
consensus_agent.py
---------------------------------------------------------------
Consensus Engine & Traceable Clinician-Facing Decision Brief.

Synthesizes all independent verification audits, adversarial challenges,
acquisition history, and evidence states into an accountable clinical decision brief.
"""
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.schemas import ConsensusResult


class ConsensusAgent(BaseAgent):
    name = "consensus_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ev = context.get("evidence", {})
        adv = context.get("adversarial", {})
        acq = context.get("acquisition", {})
        rec = acq.get("recommendation", acq) if isinstance(acq, dict) else {}
        pred = context.get("prediction", {})
        verdicts = context.get("agent_verdicts", [])

        state = ev.get("decision_state", "insufficient")
        prob = float(pred.get("pd_probability", 0.5))
        cal_conf = float(pred.get("confidence_margin", 0.5))
        has_adv_challenges = bool(adv.get("unsupported_reasoning_found", False))
        adv_challenges = adv.get("challenges", [])

        # Gather all unresolved concerns across all verifiers and adversarial critic
        unresolved_concerns: List[str] = list(adv_challenges)
        for v in verdicts:
            for c in v.get("concerns", []):
                if c not in unresolved_concerns:
                    unresolved_concerns.append(c)

        # Determine final action
        if state == "sufficient" and not has_adv_challenges:
            final_action = "present_decision"
            headline = f"Evidence is SUFFICIENT. Estimated Parkinsonian Likelihood: {prob:.2f} (Confidence: {cal_conf:.2f})."
        elif state == "sufficient" and has_adv_challenges:
            final_action = "present_decision_with_caveats"
            headline = f"Evidence is nominally sufficient, but adversarial audit noted {len(adv_challenges)} caveat(s)."
        else:
            final_action = "acquire_more_evidence"
            best_test = rec.get("assessment", "repeat_sustained_vowel")
            headline = f"Evidence is {state.upper()}. Recommend sequential acquisition: '{best_test}' before presentation."

        # Traceability Matrix
        traceability = {
            "evidence_state": state,
            "dominant_gaps": ev.get("dominant_gaps", []),
            "signal_quality": ev.get("signal_quality"),
            "predictive_entropy": pred.get("predictive_entropy"),
            "model_disagreement": pred.get("model_disagreement"),
            "phenotype_consistency": ev.get("phenotype_consistency"),
            "temporal_status": ev.get("temporal_status"),
            "verifiers_audited": [v.get("verifier_id", v.get("agent")) for v in verdicts],
            "adversarial_severity": adv.get("severity", "low"),
            "recommended_acquisition": rec.get("assessment") if final_action == "acquire_more_evidence" else None,
        }

        fallback = (
            f"CLINICIAN DECISION BRIEF: {headline}\n"
            f"- Primary Evidence State: {state.upper()}\n"
            f"- Model PD Probability: {prob:.2f} (Confidence Margin: {cal_conf:.2f})\n"
            f"- Dominant Evidentiary Gaps: {', '.join(ev.get('dominant_gaps', [])) or 'None'}\n"
            f"- Unresolved Audit Caveats: {'; '.join(unresolved_concerns) if unresolved_concerns else 'None'}\n"
            f"- Recommended Action: {final_action.upper()}."
        )

        prompt = (
            "You are a consensus neurologist synthesizing an evidence-aware decision brief for Parkinson's voice assessment. "
            f"Final Action='{final_action}', State='{state}', PD Probability={prob:.2f}, Confidence={cal_conf:.2f}. "
            f"Unresolved Concerns: {unresolved_concerns if unresolved_concerns else 'none'}. "
            "Write a concise (3-4 sentence) professional clinical summary: findings, remaining uncertainties, and actionable next step."
        )
        explanation = self.narrate(prompt, fallback)

        res = ConsensusResult(
            final_action=final_action,
            headline=headline,
            decision_state=state,
            pd_probability=round(prob, 3),
            calibrated_confidence=round(cal_conf, 3),
            unresolved_concerns=unresolved_concerns,
            decision_brief=explanation,
            traceability_matrix=traceability,
        )
        return res.to_dict()
