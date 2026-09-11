"""
cli.py
---------------------------------------------------------------
CLI Interface for Dataset Management:
- python -m src.data.cli list
- python -m src.data.cli status
- python -m src.data.cli validate <dataset_id>
- python -m src.data.cli info <dataset_id>
"""
import sys
import argparse
import json
from src.data.registry import DatasetRegistry
from src.data.assessment_capability import get_available_assessments_for_dataset
from src.data.feature_availability import get_dataset_feature_availability


def main():
    parser = argparse.ArgumentParser(description="Dataset Management CLI for Parkinson's Voice Assessment")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: list
    subparsers.add_parser("list", help="List all registered datasets")

    # Command: status
    subparsers.add_parser("status", help="Show status of all datasets")

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate a specific dataset or all datasets")
    val_parser.add_argument("dataset_id", nargs="?", default="all", help="Dataset ID to validate (default: all)")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Show detailed manifest & feature availability for a dataset")
    info_parser.add_argument("dataset_id", help="Dataset ID")

    args = parser.parse_args()

    if args.command == "list":
        datasets = DatasetRegistry.list_datasets()
        print("\nRegistered Datasets:")
        for d in datasets:
            m = DatasetRegistry.get_manifest(d)
            print(f" - {d:<32} [{m.research_role}] (Status: {m.status})")

    elif args.command == "status":
        print("\n" + "=" * 75)
        print(f"{'Dataset ID':<32} {'Research Role':<24} {'Status':<16}")
        print("=" * 75)
        for d in DatasetRegistry.list_datasets():
            m = DatasetRegistry.get_manifest(d)
            print(f"{d:<32} {m.research_role:<24} {m.status:<16}")
        print("=" * 75)

    elif args.command == "validate":
        if args.dataset_id == "all":
            reports = DatasetRegistry.validate_all()
            for d_id, rep in reports.items():
                print(f"\n--- {d_id} ---")
                print(rep.summary_text())
        else:
            rep = DatasetRegistry.validate_dataset(args.dataset_id)
            print(rep.summary_text())

    elif args.command == "info":
        m = DatasetRegistry.get_manifest(args.dataset_id)
        print("\n" + "=" * 70)
        print(f"DATASET INFO: {m.name} ({m.dataset_id})")
        print("=" * 70)
        print(f"Source URL:       {m.source_url}")
        print(f"Research Role:    {m.research_role}")
        print(f"Status:           {m.status}")
        print(f"Has Longitudinal: {m.has_longitudinal_data}")
        print(f"Has UPDRS:        {m.has_updrs}")
        print(f"Raw Path:         {m.raw_path}")
        print(f"Expected Files:   {', '.join(m.expected_files)}")
        
        # Assessments
        ass = get_available_assessments_for_dataset(args.dataset_id)
        print(f"\nSupported Assessments ({len(ass)}):")
        for a in ass:
            print(f"  - {a.assessment_id:<28} [{a.modality}] ({a.availability_status})")

        # Feature availability
        feat_map = get_dataset_feature_availability(args.dataset_id)
        avail_cnt = sum(1 for v in feat_map.values() if v.available)
        print(f"\nFeature Availability: {avail_cnt}/{len(feat_map)} tracked features present")
        for fn, fa in sorted(feat_map.items()):
            symbol = "YES" if fa.available else " NO"
            print(f"  [{symbol}] {fn:<22} ({fa.source})")
        print("=" * 70)


    else:
        parser.print_help()


if __name__ == "__main__":
    main()
