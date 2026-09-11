"""
evaluation.py
---------------------------------------------------------------
Rigorous Research Evaluation Framework.

Computes:
1. Discrimination (AUC), Calibration (Brier, ECE, Reliability Diagrams)
2. Selective Prediction (Risk-Coverage Curves)
3. Change Detection Metrics against Ground-Truth Events (Precision, Recall, F1, Specificity, FPR, FNR)
4. Unsupported-Decision Rate (UDR)
5. Sequential Information Acquisition & Efficiency Metrics
6. Multi-Policy Benchmarking (6 Acquisition Policies)
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.models import VoiceEnsemble, compute_binary_entropy, compute_expected_calibration_error
from src.features import add_phenotype_columns
from src.baseline import (
    compute_patient_baseline,
    evaluate_change,
    evaluate_naive_population_baseline,
    evaluate_simple_zscore_baseline,
)
from src.evidence import assess_evidence
from src.acquisition import (
    recommend_next_assessment,
    sample_assessment_observation,
    ASSESSMENT_CATALOG,
)
from src.policies import POLICY_REGISTRY, BaseAcquisitionPolicy


# ---------------------------------------------------------------------
# 1. Discrimination & Calibration
# ---------------------------------------------------------------------
def discrimination_and_calibration(y_true: Any, y_prob: Any) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    ece, cal_details = compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10)
    return {
        "auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "calibration_bins": cal_details,
    }


# ---------------------------------------------------------------------
# 2. Selective Prediction / Risk-Coverage
# ---------------------------------------------------------------------
def selective_risk_curve(y_true: Any, y_prob: Any, confidence: Any, coverages: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Computes selective prediction risk at varying coverage thresholds."""
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    conf_arr = np.asarray(confidence, dtype=float)

    order = np.argsort(-conf_arr)  # highest confidence first
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * len(y_true_arr))))
        idx = order[:k]
        preds = (y_prob_arr[idx] >= 0.5).astype(int)
        err = float(np.mean(preds != y_true_arr[idx]))
        acc = float(1.0 - err)
        rows.append({"coverage": round(cov, 2), "selective_risk": round(err, 4), "accuracy": round(acc, 4), "n_samples": k})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Ground-Truth Change Detection Performance (Section 10)
# ---------------------------------------------------------------------
def change_detection_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compares 4 change detection paradigms against explicit ground-truth change events:
    A. Naive Population Threshold
    B. Simple Patient Z-Score Baseline
    C. Robust Patient Baseline (Median/MAD without persistence)
    D. Proposed Longitudinal Method (Robust MAD + Minimum Effect + Persistence + Multi-Axis)
    """
    df = add_phenotype_columns(df.sort_values(["patient_id", "visit_id"])).reset_index(drop=True)
    
    methods = {
        "naive_population_threshold": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "simple_patient_zscore": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "robust_mad_baseline_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "proposed_longitudinal_method": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("visit_id").reset_index(drop=True)
        for i in range(1, len(group)):
            row = group.iloc[i]
            vid = int(row["visit_id"])
            
            # Ground truth from explicit generator column
            is_true_event = bool(row.get("known_change_event", 0) == 1 or (row.get("change_type") == "progression_jump"))
            
            # A. Naive population threshold
            naive_res = evaluate_naive_population_baseline(row, pop_threshold=0.60)
            det_a = bool(naive_res["meaningful_change"])
            
            # B. Simple z-score
            simple_res = evaluate_simple_zscore_baseline(row, group, vid, z_thresh=1.5)
            det_b = bool(simple_res["meaningful_change"])
            
            # C. Robust baseline without persistence
            baseline_c = compute_patient_baseline(group, vid)
            if baseline_c and baseline_c.has_baseline:
                max_dev = max(abs((row[col] - baseline_c.location_estimates[col]) / baseline_c.dispersion_estimates[col])
                              for col in baseline_c.location_estimates if col in row and not pd.isna(row[col]))
                det_c = max_dev >= 1.75
            else:
                det_c = False

            # D. Proposed full method
            change_d = evaluate_change(row, baseline_c, patient_history=group)
            det_d = bool(change_d["meaningful_change"])

            # Accumulate confusion counts
            for m_key, det in [("naive_population_threshold", det_a),
                               ("simple_patient_zscore", det_b),
                               ("robust_mad_baseline_only", det_c),
                               ("proposed_longitudinal_method", det_d)]:
                if is_true_event and det:
                    methods[m_key]["tp"] += 1
                elif not is_true_event and det:
                    methods[m_key]["fp"] += 1
                elif is_true_event and not det:
                    methods[m_key]["fn"] += 1
                else:
                    methods[m_key]["tn"] += 1

    results = {}
    for m_key, counts in methods.items():
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        total = tp + fp + fn + tn
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        acc = float((tp + tn) / total) if total > 0 else 0.0

        results[m_key] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": round(spec, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "accuracy": round(acc, 4),
            "counts": counts,
        }
    return results


# ---------------------------------------------------------------------
# 4. Sequential Policy Benchmarking (Section 24)
# ---------------------------------------------------------------------
def evaluate_acquisition_policies(
    ensemble: VoiceEnsemble,
    df: pd.DataFrame,
    max_steps: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs and benchmarks all 6 acquisition policies on the same patient cohort."""
    df = add_phenotype_columns(df.sort_values(["patient_id", "visit_id"])).reset_index(drop=True)
    policy_results = {}

    # Extract all testable visits
    test_cases = []
    for pid, group in df.groupby("patient_id"):
        for _, row in group.iterrows():
            test_cases.append((pid, group, int(row["visit_id"]), row))

    for pol_name, policy in POLICY_REGISTRY.items():
        rng = np.random.default_rng(seed)
        total_costs = []
        total_burdens = []
        total_assessments = []
        uncertainty_reductions = []
        unsupported_decisions = 0
        correct_final_decisions = 0
        resolved_sufficient_count = 0
        total_eval_cases = 0

        for pid, group, vid, orig_row in test_cases:
            total_eval_cases += 1
            curr_row = orig_row.copy()
            baseline = compute_patient_baseline(group, vid)
            performed: List[str] = []
            cum_cost = 0.0
            cum_burden = 0.0

            # Initial evaluation
            row_df = pd.DataFrame([curr_row])
            init_pred = ensemble.predict_summary(row_df).iloc[0].to_dict()
            init_entropy = float(init_pred["predictive_entropy"])
            current_entropy = init_entropy

            for step in range(max_steps):
                pred = ensemble.predict_summary(pd.DataFrame([curr_row])).iloc[0].to_dict()
                change_rep = evaluate_change(curr_row, baseline, patient_history=group)
                ev = assess_evidence(pred, curr_row.to_dict(), change_rep).to_dict()

                if ev["decision_state"] == "sufficient":
                    break

                acq_dec = policy.select_next(ev, pred, performed, rng=rng)
                chosen = acq_dec.assessment
                performed.append(chosen)

                obs = sample_assessment_observation(curr_row, chosen, rng=rng)
                cum_cost += obs.cost
                cum_burden += obs.burden

                # Incorporate evidence
                if "new_signal_quality" in obs.evidence_data:
                    curr_row["signal_quality"] = obs.evidence_data["new_signal_quality"]
                if "jitter_pct" in obs.evidence_data:
                    curr_row["jitter_pct"] = obs.evidence_data["jitter_pct"]
                if "shimmer" in obs.evidence_data:
                    curr_row["shimmer"] = obs.evidence_data["shimmer"]

            # Final evaluation after acquisition
            final_pred = ensemble.predict_summary(pd.DataFrame([curr_row])).iloc[0].to_dict()
            final_change = evaluate_change(curr_row, baseline, patient_history=group)
            final_ev = assess_evidence(final_pred, curr_row.to_dict(), final_change).to_dict()
            final_entropy = float(final_pred["predictive_entropy"])

            delta_h = max(0.0, init_entropy - final_entropy)
            total_costs.append(cum_cost)
            total_burdens.append(cum_burden)
            total_assessments.append(len(performed))
            uncertainty_reductions.append(delta_h)

            final_label = int(curr_row["label"])
            final_dec_pd = int(final_pred["pd_probability"] >= 0.5)

            if final_dec_pd == final_label:
                correct_final_decisions += 1

            if final_ev["decision_state"] == "sufficient":
                resolved_sufficient_count += 1
                if final_dec_pd != final_label:
                    unsupported_decisions += 1
            else:
                # Still insufficient/contradictory
                pass

        n_cases = max(1, total_eval_cases)
        n_suff = max(1, resolved_sufficient_count)
        mean_cost = float(np.mean(total_costs))
        mean_burden = float(np.mean(total_burdens))
        mean_delta_h = float(np.mean(uncertainty_reductions))
        mean_assessments = float(np.mean(total_assessments))
        acc = float(correct_final_decisions / n_cases)
        udr = float(unsupported_decisions / n_suff)

        policy_results[pol_name] = {
            "policy_name": pol_name,
            "accuracy": round(acc, 4),
            "unsupported_decision_rate": round(udr, 4),
            "avg_assessments_used": round(mean_assessments, 3),
            "avg_acquisition_cost": round(mean_cost, 3),
            "avg_acquisition_burden": round(mean_burden, 3),
            "avg_uncertainty_reduction_bits": round(mean_delta_h, 4),
            "uncertainty_reduction_per_cost": round(mean_delta_h / max(0.01, mean_cost), 4),
            "pct_cases_resolved_to_sufficient": round(resolved_sufficient_count / n_cases, 4),
        }

    return policy_results


# ---------------------------------------------------------------------
# 5. Full Dataset Framework Metric Summary
# ---------------------------------------------------------------------
def run_framework_on_dataset(ensemble: VoiceEnsemble, df: pd.DataFrame) -> pd.DataFrame:
    df = add_phenotype_columns(df.sort_values(["patient_id", "visit_id"])).reset_index(drop=True)
    rows = []
    for pid, group in df.groupby("patient_id"):
        for _, row in group.iterrows():
            baseline = compute_patient_baseline(group, row["visit_id"])
            row_df = pd.DataFrame([row])
            pred = ensemble.predict_summary(row_df).iloc[0].to_dict()
            change_report = evaluate_change(row, baseline, patient_history=group)
            evidence = assess_evidence(pred, row.to_dict(), change_report).to_dict()

            out = {
                "patient_id": pid,
                "visit_id": row["visit_id"],
                "label": int(row["label"]),
                "pd_probability": float(pred["pd_probability"]),
                "confidence_margin": float(pred["confidence_margin"]),
                "model_disagreement": float(pred["model_disagreement"]),
                "predictive_entropy": float(pred["predictive_entropy"]),
                "decision_state": evidence["decision_state"],
                "dominant_gaps": evidence.get("dominant_gaps", []),
            }
            if evidence["decision_state"] != "sufficient":
                rec = recommend_next_assessment(evidence, pred_summary=pred)
                out.update({
                    "next_assessment": rec.assessment,
                    "acq_cost": rec.cost,
                    "acq_burden": rec.burden,
                    "acq_info_gain": rec.expected_info_gain,
                })
            else:
                out.update({"next_assessment": None, "acq_cost": 0.0, "acq_burden": 0.0, "acq_info_gain": 0.0})
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_framework_metrics(framework_df: pd.DataFrame) -> Dict[str, Any]:
    n = len(framework_df)
    state_counts = framework_df["decision_state"].value_counts(normalize=True).to_dict()

    sufficient = framework_df[framework_df["decision_state"] == "sufficient"]
    if len(sufficient):
        wrong_suff = (sufficient["pd_probability"] >= 0.5).astype(int) != sufficient["label"]
        unsupported_rate_framework = float(wrong_suff.mean())
    else:
        unsupported_rate_framework = 0.0

    all_wrong = (framework_df["pd_probability"] >= 0.5).astype(int) != framework_df["label"]
    unsupported_rate_static = float(all_wrong.mean())

    avg_cost = float(framework_df["acq_cost"].mean())
    avg_burden = float(framework_df["acq_burden"].mean())
    pct_needing_more = float((framework_df["decision_state"] != "sufficient").mean())

    return {
        "n_visits": n,
        "decision_state_distribution": {k: round(v, 4) for k, v in state_counts.items()},
        "unsupported_decision_rate_framework": round(unsupported_rate_framework, 4),
        "unsupported_decision_rate_static_always_decide": round(unsupported_rate_static, 4),
        "avg_acquisition_cost_across_dataset": round(avg_cost, 4),
        "avg_acquisition_burden_across_dataset": round(avg_burden, 4),
        "pct_cases_requiring_more_evidence": round(pct_needing_more, 4),
    }


def evaluate_dataset(
    dataset_id: str,
    raw_dir: Optional[str] = None,
    ensemble: Optional[VoiceEnsemble] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a registered dataset separately according to its intended research role.
    Does not merge datasets or assume unavailable fields.
    """
    from src.data.registry import DatasetRegistry

    manifest = DatasetRegistry.get_manifest(dataset_id)
    adapter = DatasetRegistry.get_adapter(dataset_id, raw_dir=raw_dir) if raw_dir else DatasetRegistry.get_adapter(dataset_id)
    validation = adapter.validate_source()

    if not validation.is_valid:
        return {
            "dataset_id": dataset_id,
            "status": validation.status,
            "error": "Dataset not available or invalid. Run validation for details.",
            "research_role": manifest.research_role,
            "manifest": manifest.to_dict(),
        }

    raw_df = adapter.load_raw()
    can_df = adapter.to_canonical_dataframe(raw_df)

    eval_summary = {
        "dataset_id": dataset_id,
        "name": manifest.name,
        "research_role": manifest.research_role,
        "n_records": len(can_df),
        "n_subjects": int(can_df["subject_id"].nunique()) if "subject_id" in can_df.columns else 0,
        "status": "ready",
    }
    return eval_summary

