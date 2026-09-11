#!/usr/bin/env python3
"""
train.py
---------------------------------------------------------------
Trains the local, free ensemble (Logistic Regression + Random Forest +
Gradient Boosting, all scikit-learn/CPU) on the visit-level voice dataset
and saves it to models/voice_ensemble.joblib.

Usage:
    python train.py                      # uses data/patients_visits.csv
    python train.py --data path/to.csv   # use your own dataset (same schema)
    python train.py --regenerate         # regenerate synthetic data first
"""
import argparse
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.features import add_phenotype_columns
from src.models import train_and_evaluate

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patients_visits.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.regenerate or not os.path.exists(args.data):
        from data.generate_synthetic_data import generate
        print("Generating synthetic dataset...")
        generate(out_path=args.data)

    df = pd.read_csv(args.data)
    df = add_phenotype_columns(df)

    ens, metrics, train_df, test_df = train_and_evaluate(df, seed=args.seed)
    model_path = ens.save()

    print(f"Model saved to: {model_path}")
    print("Held-out (patient-level split) metrics:")
    print(json.dumps(metrics, indent=2))

    # save enriched dataset (with phenotype columns) for reuse by app/eval
    enriched_path = os.path.join(os.path.dirname(args.data), "patients_visits_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"Phenotype-enriched dataset saved to: {enriched_path}")


if __name__ == "__main__":
    main()
