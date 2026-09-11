"""
visualizations.py
---------------------------------------------------------------
Publication-Quality Research Visualizations.

Generates 10 core research figures (Section 32):
1. Calibration Curve (Reliability Diagram)
2. Selective Risk-Coverage Curve
3. Change Detection Method Comparison (Precision, Recall, F1, FPR)
4. Sequential Uncertainty (Entropy) Reduction Across Iterations
5. Acquisition Cost vs. Uncertainty Reduction Scatter/Pareto
6. Multi-Policy Benchmark Comparison (Accuracy, Cost, UDR)
7. Evidence Decision-State Distribution Across Difficulty Scenarios
8. Assessment Selection Frequency by Policy
9. Component Ablation Contribution (A0 to A8)
10. Patient Longitudinal Trajectory Case Study
"""
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "plots")


def plot_calibration_curve(cal_bins: Dict[str, Any], ece: float, out_path: Optional[str] = None):
    """1. Calibration Curve (Reliability Diagram)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    centers = cal_bins.get("bin_centers", [])
    accs = cal_bins.get("bin_accuracies", [])
    confs = cal_bins.get("bin_confidences", [])
    
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(confs, accs, "s-", color="#1f77b4", linewidth=2, markersize=7, label=f"Calibrated Ensemble (ECE = {ece:.3f})")
    ax.bar(centers, accs, width=0.08, alpha=0.25, color="#1f77b4", edgecolor="black", label="Empirical Accuracy")
    
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Reliability Diagram & Probability Calibration", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "01_calibration_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_risk_coverage_curve(rc_df: pd.DataFrame, out_path: Optional[str] = None):
    """2. Selective Prediction / Risk-Coverage Curve."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(rc_df["coverage"], rc_df["selective_risk"], "o-", color="#d62728", linewidth=2.2, markersize=6, label="Selective Risk (Error Rate)")
    ax.plot(rc_df["coverage"], rc_df["accuracy"], "s--", color="#2ca02c", linewidth=1.8, markersize=5, label="Selective Accuracy")

    ax.set_xlabel("Coverage Ratio (Retained Samples)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rate", fontsize=11, fontweight="bold")
    ax.set_title("Selective Risk vs. Coverage (Uncertainty Deferral)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center left")

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "02_risk_coverage_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_change_detection_comparison(cd_results: Dict[str, Any], out_path: Optional[str] = None):
    """3. Change Detection Method Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    methods = list(cd_results.keys())
    labels = [m.replace("_", " ").title() for m in methods]
    
    f1_scores = [cd_results[m]["f1_score"] for m in methods]
    precision = [cd_results[m]["precision"] for m in methods]
    recall = [cd_results[m]["recall"] for m in methods]
    fpr = [cd_results[m]["false_positive_rate"] for m in methods]

    x = np.arange(len(methods))
    width = 0.20

    ax.bar(x - 1.5*width, precision, width, label="Precision", color="#1f77b4")
    ax.bar(x - 0.5*width, recall, width, label="Recall", color="#2ca02c")
    ax.bar(x + 0.5*width, f1_scores, width, label="F1 Score", color="#ff7f0e")
    ax.bar(x + 1.5*width, fpr, width, label="False Positive Rate", color="#d62728")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Patient-Specific Change Detection vs. Baseline Approaches", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "03_change_detection_comparison.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_uncertainty_reduction_iterations(iterations_data: List[Dict[str, Any]], out_path: Optional[str] = None):
    """4. Sequential Uncertainty Reduction Across Iterations."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    iters = [it["iteration"] for it in iterations_data]
    entropies = [it["prediction"]["predictive_entropy"] for it in iterations_data]
    disagreements = [it["prediction"]["model_disagreement"] for it in iterations_data]

    ax.plot(iters, entropies, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Shannon Predictive Entropy (bits)")
    ax.plot(iters, disagreements, "s--", color="#e377c2", linewidth=2.0, markersize=7, label="Model Disagreement (std)")

    ax.set_xlabel("Sequential Assessment Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Uncertainty Metric", fontsize=11, fontweight="bold")
    ax.set_title("Iterative Uncertainty Reduction Trajectory", fontsize=12, fontweight="bold")
    ax.set_xticks(iters)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "04_uncertainty_reduction_trajectory.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cost_vs_uncertainty_reduction(trajectories: List[Dict[str, Any]], out_path: Optional[str] = None):
    """5. Acquisition Cost vs. Uncertainty Reduction."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    costs = [t["total_cost"] for t in trajectories]
    reductions = [t["absolute_uncertainty_reduction"] for t in trajectories]
    burdens = [t["total_burden"] for t in trajectories]

    sc = ax.scatter(costs, reductions, c=burdens, cmap="viridis", s=70, alpha=0.85, edgecolors="black")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Total Patient Burden", fontsize=10, fontweight="bold")

    ax.set_xlabel("Total Acquisition Cost ($ arbitrary units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Shannon Entropy Reduction (bits)", fontsize=11, fontweight="bold")
    ax.set_title("Sequential Acquisition Cost vs. Information Gain", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "05_cost_vs_uncertainty_reduction.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_policy_comparison(policy_metrics: Dict[str, Any], out_path: Optional[str] = None):
    """6. Multi-Policy Benchmark Comparison."""
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    policies = list(policy_metrics.keys())
    labels = [p.replace("_", " ").title() for p in policies]

    accs = [policy_metrics[p]["accuracy"] for p in policies]
    udrs = [policy_metrics[p]["unsupported_decision_rate"] for p in policies]
    costs = [policy_metrics[p]["avg_acquisition_cost"] for p in policies]

    x = np.arange(len(policies))
    width = 0.28

    r1 = ax1.bar(x - width/2, accs, width, label="Decision Accuracy", color="#2ca02c", alpha=0.85)
    r2 = ax1.bar(x + width/2, udrs, width, label="Unsupported Decision Rate", color="#d62728", alpha=0.85)

    ax1.set_ylabel("Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Acquisition Policy Benchmark: Accuracy, Safety (UDR), and Cost", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0.0, 1.1)

    # Secondary axis for Cost
    ax2 = ax1.twinx()
    r3 = ax2.plot(x, costs, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Average Cost")
    ax2.set_ylabel("Average Cost", color="#1f77b4", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, max(costs)*1.3 if costs else 1.0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "06_policy_comparison.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scenario_state_distributions(scenario_summaries: Dict[str, Any], out_path: Optional[str] = None):
    """7. Decision-State Distribution Across Difficulty Scenarios."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    scenarios = list(scenario_summaries.keys())
    
    sufficient = [scenario_summaries[s]["decision_state_distribution"].get("sufficient", 0.0) for s in scenarios]
    insufficient = [scenario_summaries[s]["decision_state_distribution"].get("insufficient", 0.0) for s in scenarios]
    contradictory = [scenario_summaries[s]["decision_state_distribution"].get("contradictory", 0.0) for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.55

    ax.bar(x, sufficient, width, label="Sufficient Evidence", color="#2ca02c", alpha=0.85)
    ax.bar(x, insufficient, width, bottom=sufficient, label="Insufficient Evidence", color="#ff7f0e", alpha=0.85)
    bottom_contra = np.array(sufficient) + np.array(insufficient)
    ax.bar(x, contradictory, width, bottom=bottom_contra, label="Contradictory Evidence", color="#d62728", alpha=0.85)

    ax.set_ylabel("Proportion of Patient Visits", fontsize=11, fontweight="bold")
    ax.set_title("Evidence State Distribution Across Dataset Difficulty Scenarios", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in scenarios], fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "07_decision_state_distributions.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_assessment_frequencies(freq_dict: Dict[str, int], out_path: Optional[str] = None):
    """8. Assessment Selection Frequency."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    tests = list(freq_dict.keys())
    counts = list(freq_dict.values())
    labels = [t.replace("_", " ").title() for t in tests]

    bars = ax.barh(labels, counts, color="#17becf", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Selection Frequency (Count)", fontsize=11, fontweight="bold")
    ax.set_title("Next-Best Assessment Selection Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(counts)*0.01, bar.get_y() + bar.get_height()/2, f"{int(w)}", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "08_assessment_selection_frequency.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ablation_performance(ablation_data: Dict[str, Any], out_path: Optional[str] = None):
    """9. Component Ablation Performance Contribution (A0 to A8)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    levels = list(ablation_data.keys())
    labels = [ablation_data[k]["level"] for k in levels]

    accs = [ablation_data[k]["accuracy"] for k in levels]
    udrs = [ablation_data[k]["unsupported_decision_rate"] for k in levels]
    deferral = [ablation_data[k]["actionable_deferral_rate"] for k in levels]

    x = np.arange(len(levels))
    ax.plot(x, accs, "o-", color="#2ca02c", linewidth=2.5, markersize=8, label="Accuracy (on Decided Cases)")
    ax.plot(x, udrs, "s-", color="#d62728", linewidth=2.5, markersize=8, label="Unsupported Decision Rate (UDR)")
    ax.plot(x, deferral, "^--", color="#ff7f0e", linewidth=1.8, markersize=7, label="Actionable Deferral / Acquisition Rate")

    ax.set_xlabel("Ablation Level (A0 = Static Single Classifier ... A8 = Full Framework)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Incremental Component Ablation Study (A0 to A8)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right")

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "09_ablation_performance.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_longitudinal_case_study(patient_df: pd.DataFrame, out_path: Optional[str] = None):
    """10. Patient Longitudinal Trajectory Case Study."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)
    vids = patient_df["visit_id"].values
    
    # Phenotype trajectory
    ax1.plot(vids, patient_df["phenotype_vocal_stability"], "o-", label="Vocal Stability", color="#1f77b4")
    ax1.plot(vids, patient_df["phenotype_amplitude_variation"], "s-", label="Amplitude Variation", color="#ff7f0e")
    ax1.plot(vids, patient_df["phenotype_noise_characteristics"], "^-", label="Noise Characteristics", color="#2ca02c")
    ax1.plot(vids, patient_df["phenotype_overall_phenotype_severity"], "k--", linewidth=2, label="Overall Phenotype Severity")

    ax1.set_ylabel("Normalized Severity", fontsize=10, fontweight="bold")
    ax1.set_title(f"Longitudinal Trajectory Case Study: Patient {patient_df['patient_id'].iloc[0]}", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    # UPDRS & Signal Quality
    ax2.plot(vids, patient_df["updrs_like"], "d-", color="#9467bd", label="UPDRS-like Score (Clinician Scale)")
    ax2.set_xlabel("Visit Number", fontsize=10, fontweight="bold")
    ax2.set_ylabel("UPDRS Score", color="#9467bd", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(vids, patient_df["signal_quality"], "x--", color="#8c564b", label="Signal Quality")
    ax2_twin.set_ylabel("Signal Quality (0-1)", color="#8c564b", fontsize=10, fontweight="bold")
    ax2_twin.set_ylim(0, 1.1)

    plt.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "10_longitudinal_case_study.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
