#!/usr/bin/env python3
"""
run_evaluation.py
---------------------------------------------------------------
Runs the full evaluation suite described in the abstract:
  - Discrimination (AUC) + calibration (Brier score, ECE) for the static
    ensemble classifier baseline
  - Selective risk / risk-coverage curve (uncertainty-aware baseline)
  - Evidence-aware framework metrics: decision-state distribution,
    unsupported-decision rate (framework vs. always-decide static
    baseline), average acquisition cost/burden, % cases needing more evidence
  - Patient-specific change-detection performance vs. a naive
    population-threshold baseline (ablation)

Usage:
    python run_evaluation.py
    python run_evaluation.py --save outputs/evaluation_report.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.features import add_phenotype_columns
from src.models import VoiceEnsemble, train_and_evaluate
from src.evaluation import (
    discrimination_and_calibration, selective_risk_curve,
    run_framework_on_dataset, summarize_framework_metrics,
    change_detection_metrics,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patients_visits.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--save", default=os.path.join(os.path.dirname(__file__), "outputs", "evaluation_report.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_df = pd.read_csv(args.data)
    df = add_phenotype_columns(raw_df)

    print("Training ensemble (patient-level split) for evaluation...")
    ens, base_metrics, train_df, test_df = train_and_evaluate(df, seed=args.seed)

    pred = ens.predict_summary(test_df)

    print("\n[1] STATIC CLASSIFIER: Discrimination & Calibration")
    disc_cal = discrimination_and_calibration(test_df["label"], pred["pd_probability"])
    print(json.dumps(disc_cal, indent=2))

    print("\n[2] UNCERTAINTY-AWARE BASELINE: Selective risk / risk-coverage")
    src_df = selective_risk_curve(test_df["label"], pred["pd_probability"], pred["confidence_margin"])
    print(src_df.to_string(index=False))

    print("\n[3] EVIDENCE-AWARE FRAMEWORK: decision states, unsupported-decision rate, acquisition cost/burden")
    framework_df = run_framework_on_dataset(ens, raw_df)
    fw_metrics = summarize_framework_metrics(framework_df)
    print(json.dumps(fw_metrics, indent=2, default=str))

    print("\n[4] PATIENT-SPECIFIC CHANGE DETECTION vs. naive population threshold (ablation)")
    cd_metrics = change_detection_metrics(raw_df)
    print(json.dumps(cd_metrics, indent=2, default=str))

    report = {
        "base_model_metrics": base_metrics,
        "discrimination_and_calibration": disc_cal,
        "selective_risk_curve": src_df.to_dict(orient="records"),
        "framework_metrics": fw_metrics,
        "change_detection_metrics": cd_metrics,
    }
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull evaluation report saved to: {args.save}")


if __name__ == "__main__":
    main()
