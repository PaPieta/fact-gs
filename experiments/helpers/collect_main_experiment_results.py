#!/usr/bin/env python3

"""Aggregate PSNR/SSIM across all runs of main_experiment.sh."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Root directory containing recon models grouped by domain/split.",
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
    records = collect_metrics_records(args.model_root)
    if not records:
        raise SystemExit(
            f"No *_metrics_final.yml files found under {args.model_root}"
        )

    df = pd.DataFrame(records)
    df.sort_values(["domain", "data_name", "split"], inplace=True)

    psnr_pivot = pivot_metric(df, value_column="psnr_3d", prefix="psnr")
    ssim_pivot = pivot_metric(df, value_column="ssim_3d", prefix="ssim")
    combined = pd.concat([psnr_pivot, ssim_pivot], axis=1)

    psnr_cols = [col for col in combined.columns if col.startswith("psnr_")]
    ssim_cols = [col for col in combined.columns if col.startswith("ssim_")]
    combined["avg_psnr"] = combined[psnr_cols].mean(axis=1, skipna=True)
    combined["avg_ssim"] = combined[ssim_cols].mean(axis=1, skipna=True)

    combined.reset_index(inplace=True)

    print(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    # Print aggregate averages across all datasets
    avg_summary = combined[psnr_cols + ssim_cols].mean(numeric_only=True)
    print("\nColumn-wise averages:")
    print(avg_summary.to_string())
    print(f"\nSaved metrics to {args.output_csv}")


def collect_metrics_records(model_root: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    pattern = "*_metrics_final.yml"
    for metrics_file in sorted(model_root.rglob(pattern)):
        rel = metrics_file.relative_to(model_root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        domain = parts[0]
        split = parts[1] if len(parts) >= 3 else "unknown_split"
        data_name = metrics_file.stem.replace("_metrics_final", "")

        with open(metrics_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        records.append(
            {
                "domain": domain,
                "split": split,
                "data_name": data_name,
                "psnr_3d": data.get("psnr_3d"),
                "ssim_3d": data.get("ssim_3d"),
            }
        )
    return records


def pivot_metric(df: pd.DataFrame, value_column: str, prefix: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["domain", "data_name"],
        columns="split",
        values=value_column,
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=1)
    pivot.columns = [f"{prefix}_{col}" for col in pivot.columns]
    return pivot


if __name__ == "__main__":
    main()
