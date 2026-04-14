#!/usr/bin/env python3

"""Collect metrics for coral scaling studies (fixed-iterations or SSIM target)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

GAUSSIAN_COUNTS = {
    "coral256_cone": 50_000,
    "coral384_cone": 75_000,
    "coral512_cone": 100_000,
    "coral768_cone": 150_000,
    "coral1k_cone": 200_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Directory containing scaling study model outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.study_root)
    if not rows:
        raise SystemExit(f"No metrics found under {args.study_root}")

    df = pd.DataFrame(rows)
    df.sort_values("dataset_name", inplace=True)
    print(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved coral scaling study metrics to {args.output_csv}")


def collect_rows(study_root: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for dataset_name, gaussian_count in GAUSSIAN_COUNTS.items():
        metrics_path = study_root / f"{dataset_name}_metrics_final.yml"
        if not metrics_path.exists():
            print(f"Missing metrics for {dataset_name}: {metrics_path}")
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        training_time = data.get("time_training_seconds")
        if training_time is None:
            print(f"Incomplete metrics for {dataset_name}")
            continue
        row = {
            "dataset_name": dataset_name,
            "gaussian_count": gaussian_count,
            "training_time_seconds": float(training_time),
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
