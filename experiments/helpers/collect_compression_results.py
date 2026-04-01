#!/usr/bin/env python3

"""Merge baseline compression stats with Gaussian-splat metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing *_metrics_final.yml files.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="Path to baseline compression CSV.",
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
    if not args.baseline_csv.exists():
        raise SystemExit(f"Missing baseline CSV: {args.baseline_csv}")
    baseline_df = pd.read_csv(args.baseline_csv)

    metrics_rows = gather_method_metrics(args.results_root)
    if not metrics_rows:
        raise SystemExit(f"No *_metrics_final.yml files found under {args.results_root}")
    method_df = pd.DataFrame(metrics_rows)

    combined = baseline_df.merge(method_df, on="instance", how="left")
    print(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    print(f"\nSaved combined CSV to {args.output_csv}")


def gather_method_metrics(results_root: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for metrics_file in sorted(results_root.glob("*_metrics_final.yml")):
        instance_name = metrics_file.name.replace("_metrics_final.yml", "")
        with open(metrics_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        row = {
            "instance": instance_name,
            "gs_psnr_3d": data.get("psnr_3d"),
            "gs_ssim_3d": data.get("ssim_3d"),
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
