#!/usr/bin/env python3
"""
app.py
---------------------------------------------------------------
Evidence-Aware Sequential Decision Support Dashboard for Parkinson's Voice Assessment.

100% local, transparent, and reproducible:
- Calibrated ML ensemble with predictive entropy & model disagreement
- Structured speech phenotype & robust patient baseline change validation
- Formal Next-Best Assessment Policy (EIG & Cost-Burden Utility)
- Multi-Agent Independent Verification & Adversarial Auditing
- Traceable Clinician-Facing Decision Briefs & Provenance Matrices
"""
import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.features import add_phenotype_columns, compute_signal_quality_metrics
from src.models import VoiceEnsemble, train_and_evaluate
from src.agents.orchestrator import DecisionOrchestrator
from src.agents.base_agent import ollama_available
from src.acquisition import ASSESSMENT_CATALOG

from src.data.registry import DatasetRegistry

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patients_visits.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "voice_ensemble.joblib")

st.set_page_config(page_title="Parkinson's Evidence-Aware Decision Support", layout="wide", page_icon="🧠")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        from data.generate_synthetic_data import generate
        generate(out_path=path, scenario="moderate")
    df = pd.read_csv(path)
    return add_phenotype_columns(df)


@st.cache_resource
def load_or_train_model(df: pd.DataFrame) -> VoiceEnsemble:
    if os.path.exists(MODEL_PATH):
        try:
            return VoiceEnsemble.load(MODEL_PATH)
        except Exception:
            pass
    ens, _, _, _ = train_and_evaluate(df, seed=42)
    ens.save(MODEL_PATH)
    return ens


def main():
    st.title("🧠 Evidence-Aware Sequential Decision Support — Parkinson's Voice Assessment")
    st.caption("Rigorous Clinical Decision-Support Framework with Adaptive Information Acquisition, Patient-Specific Baselines, and Multi-Agent Verification.")

    df = load_data(DATA_PATH)
    ens = load_or_train_model(df)

    llm_up = ollama_available()
    with st.sidebar:
        st.header("⚙️ Configuration & Patient Selection")
        st.write(f"Local LLM (Ollama): {'✅ Connected' if llm_up else 'ℹ️ Offline (Deterministic Rule Engine Active)'}")
        use_llm = st.checkbox("Enable LLM Narrative Synthesis", value=llm_up)
        llm_model = st.text_input("Ollama Model Name", value="llama3.2:1b")

        st.markdown("---")
        st.subheader("📁 Dataset Status")
        manifests = DatasetRegistry.get_all_manifests()
        for d_id, m in manifests.items():
            status_icon = "🟢 Ready" if m.status in ("ready", "validated") else "⚪ Not Downloaded"
            st.caption(f"**{m.name}**: {status_icon}")

        st.markdown("---")
        patients = sorted(df["patient_id"].unique())
        selected_pid = st.selectbox("Select Patient", patients, index=patients.index("P001") if "P001" in patients else 0)
        patient_history = df[df["patient_id"] == selected_pid].sort_values("visit_id").reset_index(drop=True)
        visits = patient_history["visit_id"].tolist()
        selected_vid = st.selectbox("Select Current Visit", visits, index=len(visits) - 1)
        
        st.markdown("---")
        max_budget = st.slider("Max Acquisition Cost Budget", 0.20, 2.50, 1.80, 0.10)
        run_btn = st.button("▶ Run Evidence-Aware Assessment", type="primary", use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Sequential Decision Pipeline",
        "📈 Longitudinal History & Baseline",
        "🧭 Feasible Assessment Space",
        "📊 Evaluation & Benchmark Suite",
        "📁 Real Dataset Status & Integration",
    ])


    # TAB 2: Longitudinal Record
    with tab2:
        st.subheader(f"Longitudinal Clinical Record — Patient {selected_pid}")
        cols = ["visit_id", "visit_date", "group", "updrs_like", "signal_quality",
                "phenotype_overall_phenotype_severity", "known_change_event", "label"]
        available_cols = [c for c in cols if c in patient_history.columns]
        st.dataframe(patient_history[available_cols], use_container_width=True)

        st.markdown("#### Speech Phenotype Trajectory Across Visits")
        phen_axes = [c for c in ["phenotype_vocal_stability", "phenotype_amplitude_variation",
                                 "phenotype_noise_characteristics", "phenotype_nonlinear_dynamics"] if c in patient_history.columns]
        if phen_axes:
            st.line_chart(patient_history.set_index("visit_id")[phen_axes])

    # TAB 3: Assessment Space Catalog
    with tab3:
        st.subheader("Predefined Feasible Information-Acquisition Catalog")
        catalog_rows = []
        for a_id, defn in ASSESSMENT_CATALOG.items():
            catalog_rows.append({
                "Assessment ID": defn.assessment_id,
                "Name": defn.name,
                "Evidence Modality": defn.evidence_type,
                "Cost ($)": defn.cost,
                "Patient Burden": defn.burden,
                "Duration (min)": defn.expected_duration_minutes,
                "Targeted Evidentiary Gaps": ", ".join(defn.target_gaps.keys()),
                "Clinical Description": defn.description,
            })
        st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True)

    # TAB 1: Main Case Execution
    with tab1:
        if run_btn:
            orch = DecisionOrchestrator(ens, use_llm=use_llm, llm_model=llm_model)
            with st.spinner("Executing multi-agent evidence-aware sequential decision process..."):
                result = orch.run_case(patient_history, selected_vid, cost_budget=max_budget)

            final_it = result["iterations"][-1]
            state = final_it["evidence"]["decision_state"]
            state_color = {"sufficient": "green", "insufficient": "orange", "contradictory": "red"}[state]

            st.markdown(f"### Assessment Outcome for **{selected_pid} (Visit {selected_vid})**")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("PD Probability", f"{final_it['prediction']['pd_probability']:.2f}")
            m2.metric("Confidence Margin", f"{final_it['prediction']['confidence_margin']:.2f}")
            m3.metric("Predictive Entropy", f"{final_it['prediction']['predictive_entropy']:.2f} bits")
            m4.metric("Model Disagreement", f"{final_it['prediction']['model_disagreement']:.2f}")
            m5.markdown(f"**Evidence State:** :{state_color}[{state.upper()}]")

            # Dominant Gaps Banner
            gaps = final_it["evidence"].get("dominant_gaps", [])
            if gaps:
                st.warning(f"**Active Evidentiary Gaps Identified:** {', '.join(gaps)}")
            else:
                st.success("✅ **No Dominant Evidentiary Gaps:** Sufficient evidence base established.")

            # Sequential Trajectory
            st.markdown("---")
            st.subheader("🔁 Iterative Evidence Gathering Trajectory")
            for i, it in enumerate(result["iterations"]):
                with st.expander(
                    f"Iteration {i}: Decision State = {it['evidence']['decision_state'].upper()} | Entropy = {it['prediction']['predictive_entropy']:.2f} bits | Cost = ${it['cumulative_cost']:.2f}",
                    expanded=(i == len(result["iterations"]) - 1),
                ):
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**Evidence Report:**")
                        st.json(it["evidence"], expanded=False)
                    with c_right:
                        if "acquisition" in it:
                            rec = it["acquisition"]["recommendation"]
                            st.info(
                                f"**Recommended Assessment:** `{rec['assessment']}`\n\n"
                                f"- **Expected Info Gain (EIG):** {rec['expected_info_gain']:.3f} bits\n"
                                f"- **Cost:** {rec['cost']:.2f} | **Burden:** {rec['burden']:.2f} | **Utility:** {rec['utility']:.3f}\n\n"
                                f"- **Rationale:** {it['acquisition']['explanation']}"
                            )
                        if "assessment_result" in it:
                            obs = it["assessment_result"]
                            st.success(f"**Acquired Evidence Observation:**\n\n`{json.dumps(obs['evidence_data'])}`")

            # Multi-Agent Verification
            st.markdown("---")
            st.subheader("🕵️ Independent Multi-Agent Verification Layer")
            v_cols = st.columns(5)
            verifs = [
                ("Signal Quality", result["verification"]["signal_quality"]),
                ("Predictive Reliability", result["verification"]["reliability"]),
                ("Phenotype Concordance", result["verification"]["phenotype"]),
                ("Temporal Baseline", result["verification"]["temporal"]),
                ("Acquisition Verifier", result["verification"]["acquisition_verifier"]),
            ]
            for col, (title, v_res) in zip(v_cols, verifs):
                with col:
                    st.markdown(f"**{title}**")
                    status_emoji = "✅" if v_res.get("passed") else "⚠️"
                    st.write(f"Verdict: {status_emoji} `{v_res.get('verdict', 'unknown')}`")
                    st.caption(v_res.get("explanation", ""))

            # Adversarial Critic
            st.markdown("---")
            st.subheader("⚔️ Adversarial Critic Audit")
            adv = result["adversarial"]
            if adv["unsupported_reasoning_found"]:
                st.warning(f"**Adversarial Critic Caveats ({adv.get('severity', 'medium').upper()} Severity):**\n\n{adv['explanation']}")
            else:
                st.success(f"**Adversarial Critic Clearance:** {adv['explanation']}")

            # Final Decision Brief
            st.markdown("---")
            st.subheader("📋 Traceable Clinician-Facing Decision Brief")
            final_act = result["consensus"]["final_action"]
            st.markdown(f"**Action Directive: `{final_act.upper()}`**")
            st.info(result["consensus"]["decision_brief"])

            with st.expander("🔍 Traceability Matrix & Full Provenance JSON"):
                st.json(result, expanded=False)
        else:
            st.info("Select a patient and visit in the sidebar, then click **▶ Run Evidence-Aware Assessment**.")

    # TAB 4: Evaluation Suite
    with tab4:
        st.subheader("📊 Research Evaluation & Benchmark Suite")
        st.caption("Inspect saved research figures and rerun benchmarks directly.")
        
        plot_files = [
            ("Reliability Diagram & Calibration", "01_calibration_curve.png"),
            ("Selective Risk-Coverage Curve", "02_risk_coverage_curve.png"),
            ("Change Detection Method Benchmark", "03_change_detection_comparison.png"),
            ("Uncertainty Reduction Trajectory", "04_uncertainty_reduction_trajectory.png"),
            ("Acquisition Cost vs. Information Gain", "05_cost_vs_uncertainty_reduction.png"),
            ("Acquisition Policy Benchmark", "06_policy_comparison.png"),
            ("Scenario Decision-State Distributions", "07_decision_state_distributions.png"),
            ("Assessment Selection Frequency", "08_assessment_selection_frequency.png"),
            ("Component Ablation (A0 to A8)", "09_ablation_performance.png"),
            ("Patient Longitudinal Trajectory Case Study", "10_longitudinal_case_study.png"),
        ]

        plots_dir = os.path.join(os.path.dirname(__file__), "results", "plots")
        for title, fname in plot_files:
            p_path = os.path.join(plots_dir, fname)
            if os.path.exists(p_path):
                st.markdown(f"#### {title}")
                st.image(p_path, use_column_width=True)

    # TAB 5: Real Dataset Status & Integration
    with tab5:
        st.subheader("📁 Real UCI Datasets Integration Status")
        st.caption("Inspect readiness, download status, and validation for all 4 UCI Parkinson's benchmark datasets.")

        manifests = DatasetRegistry.get_all_manifests()
        for d_id, m in manifests.items():
            with st.expander(f"{m.name} ({d_id}) — Status: {m.status.upper()}", expanded=(d_id == "synthetic")):
                colA, colB = st.columns(2)
                with colA:
                    st.write(f"**Research Role**: `{m.research_role}`")
                    st.write(f"**Source URL**: [{m.source_url}]({m.source_url})")
                    st.write(f"**License**: {m.license_info}")
                with colB:
                    st.write(f"**Has Longitudinal**: {'Yes' if m.has_longitudinal_data else 'No'}")
                    st.write(f"**Has UPDRS**: {'Yes' if m.has_updrs else 'No'}")
                    st.write(f"**Expected Files**: `{', '.join(m.expected_files)}`")

                if st.button(f"Validate {d_id}", key=f"val_{d_id}"):
                    val_rep = DatasetRegistry.validate_dataset(d_id)
                    if val_rep.is_valid:
                        st.success(f"Valid! {val_rep.n_records} records across {val_rep.n_subjects} subjects.")
                    else:
                        st.warning(f"Status: {val_rep.status}. Files missing: {val_rep.files_missing}")
                        if val_rep.warnings:
                            st.info("\n".join(val_rep.warnings))


if __name__ == "__main__":

    main()
