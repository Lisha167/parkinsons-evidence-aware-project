#!/usr/bin/env python3
"""
run_case.py
---------------------------------------------------------------
Runs the FULL evidence-aware, multi-agent sequential decision pipeline for
one patient visit and prints a traceable clinician-facing decision brief.

Usage:
    python run_case.py --patient P001 --visit 3
    python run_case.py --patient P001 --visit 3 --no-llm   # force rule-based only
    python run_case.py --list-patients
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.features import add_phenotype_columns
from src.models import VoiceEnsemble
from src.agents.orchestrator import DecisionOrchestrator

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patients_visits.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "voice_ensemble.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--visit", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="disable local LLM narration, rule-based only")
    ap.add_argument("--llm-model", default="llama3.2:1b")
    ap.add_argument("--list-patients", action="store_true")
    ap.add_argument("--save", default=None, help="path to save full JSON result")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = add_phenotype_columns(df)

    if args.list_patients:
        for pid, group in df.groupby("patient_id"):
            print(f"{pid}: visits {sorted(group['visit_id'].tolist())} "
                  f"(label={'PD' if group['label'].iloc[0]==1 else 'healthy'})")
        return

    if not os.path.exists(args.model):
        print("No trained model found. Run `python train.py` first.")
        sys.exit(1)

    ens = VoiceEnsemble.load(args.model)

    if args.patient is None:
        # default to most interesting case: a PD patient's last visit
        pd_patients = df[df["label"] == 1]["patient_id"].unique()
        args.patient = pd_patients[0]
    patient_history = df[df["patient_id"] == args.patient].sort_values("visit_id").reset_index(drop=True)
    if patient_history.empty:
        print(f"No such patient: {args.patient}")
        sys.exit(1)
    if args.visit is None:
        args.visit = int(patient_history["visit_id"].max())

    orch = DecisionOrchestrator(ens, use_llm=not args.no_llm, llm_model=args.llm_model)
    result = orch.run_case(patient_history, args.visit)

    print("=" * 70)
    print(f"CASE: Patient {result['patient_id']}  |  Visit {result['visit_id']}")
    print("=" * 70)
    print(f"Iterations of evidence-gathering: {len(result['iterations'])}")
    for i, it in enumerate(result["iterations"]):
        print(f"  [{i}] decision_state={it['evidence']['decision_state']}  "
              f"pd_prob={it['prediction']['pd_probability']:.2f}  "
              f"confidence={it['prediction']['confidence_margin']:.2f}  "
              f"disagreement={it['prediction']['model_disagreement']:.2f}")
        if "acquisition" in it:
            print(f"      -> next assessment recommended: "
                  f"{it['acquisition']['recommendation']['assessment']}")
    print("-" * 70)
    print("VERIFICATION AGENTS:")
    for k, v in result["verification"].items():
        print(f"  [{k}] {v['verdict'] if 'verdict' in v else ''}: {v['explanation']}")
    print("-" * 70)
    print("ADVERSARIAL REVIEW:")
    print(f"  {result['adversarial']['explanation']}")
    print("-" * 70)
    print("ACQUISITION RECOMMENDATION (final state):")
    rec = result["acquisition"]["recommendation"]
    print(f"  {rec['assessment']}: {result['acquisition']['explanation']}")
    print("-" * 70)
    print("FINAL CONSENSUS DECISION BRIEF:")
    print(f"  [{result['consensus']['final_action']}]")
    print(f"  {result['consensus']['decision_brief']}")
    print("=" * 70)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Full JSON result saved to {args.save}")


if __name__ == "__main__":
    main()
