#!/usr/bin/env python3

"""Collect normalized SSIM metrics for the Gaussian count study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Root directory containing <data_name>/<gaussian_count> results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_metrics(args.study_root)
    if not records:
        raise SystemExit(f"No metrics found in {args.study_root}")

    df = pd.DataFrame(records)
    df["relative_ssim"] = df["ssim_3d"] / df.groupby("data_name")["ssim_3d"].transform(
        "max"
    )
    df["relative_ssim"] = df["relative_ssim"] * 100.0

    pivot = df.pivot_table(
        index="data_name",
        columns="gaussian_count",
        values="relative_ssim",
        aggfunc="max",
    )
    pivot.sort_index(axis=1, inplace=True)

    print(pivot)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.output_csv)
    print(f"\nSaved Gaussian study summary to {args.output_csv}")


def collect_metrics(study_root: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for data_dir in sorted(d for d in study_root.iterdir() if d.is_dir()):
        data_name = data_dir.name
        for gaussian_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            try:
                gaussian_count = int(gaussian_dir.name)
            except ValueError:
                continue
            metrics_file = gaussian_dir / f"{gaussian_dir.name}_metrics_final.yml"
            if not metrics_file.exists():
                continue
            with open(metrics_file, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            ssim = data.get("ssim_3d")
            if ssim is None:
                continue
            records.append(
                {
                    "data_name": data_name,
                    "gaussian_count": gaussian_count,
                    "ssim_3d": float(ssim),
                }
            )
    return records


if __name__ == "__main__":
    main()
