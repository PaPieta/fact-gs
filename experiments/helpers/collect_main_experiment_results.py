#!/usr/bin/env python3

"""Aggregate PSNR/SSIM across all runs of main_experiment.sh."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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
    parser.add_argument(
        "--metrics-suffix",
        type=str,
        default="final",
        help="Suffix used in *_metrics_<suffix>.yml files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_metrics_records(args.model_root, args.metrics_suffix)
    if not records:
        raise SystemExit(
            f"No *_metrics_{args.metrics_suffix}.yml files found under {args.model_root}"
        )

    df = pd.DataFrame(records)
    df.sort_values(["domain", "data_name", "split"], inplace=True)

    psnr_pivot = pivot_metric(df, value_column="psnr_3d", prefix="psnr")
    ssim_pivot = pivot_metric(df, value_column="ssim_3d", prefix="ssim")
    time_pivot = pivot_metric(
        df, value_column="training_time_seconds", prefix="time"
    )

    pivot_frames = [psnr_pivot, ssim_pivot]
    if not time_pivot.empty:
        pivot_frames.append(time_pivot)
    combined = pd.concat(pivot_frames, axis=1)

    psnr_cols = [col for col in combined.columns if col.startswith("psnr_")]
    ssim_cols = [col for col in combined.columns if col.startswith("ssim_")]
    time_cols = [col for col in combined.columns if col.startswith("time_")]
    summary_cols = psnr_cols + ssim_cols + time_cols
    if summary_cols:
        column_means = combined[summary_cols].mean(axis=0, skipna=True)
    else:
        column_means = pd.Series(dtype=float)

    combined.reset_index(inplace=True)

    avg_row = {"domain": "ALL", "data_name": "ALL_DATA"}
    for col, value in column_means.items():
        avg_row[col] = value
    combined = pd.concat([combined, pd.DataFrame([avg_row])], ignore_index=True)

    print(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    # Print aggregate averages across all datasets
    avg_summary = column_means
    print("\nColumn-wise averages:")
    print(avg_summary.to_string())
    print(f"\nSaved metrics to {args.output_csv}")


def collect_metrics_records(
    model_root: Path, metrics_suffix: str
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pattern = f"*_metrics_{metrics_suffix}.yml"
    suffix_tag = f"_metrics_{metrics_suffix}"
    for metrics_file in sorted(model_root.rglob(pattern)):
        rel = metrics_file.relative_to(model_root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        domain = parts[0]
        split = parts[1] if len(parts) >= 3 else "unknown_split"
        stem = metrics_file.stem
        if stem.endswith(suffix_tag):
            data_name = stem[: -len(suffix_tag)]
        else:
            data_name = stem

        with open(metrics_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        training_time = data.get("time_training_seconds")
        if training_time is None:
            training_time = data.get("training_time_seconds")

        records.append(
            {
                "domain": domain,
                "split": split,
                "data_name": data_name,
                "psnr_3d": data.get("psnr_3d"),
                "ssim_3d": data.get("ssim_3d"),
                "training_time_seconds": training_time,
            }
        )
    return records


def pivot_metric(df: pd.DataFrame, value_column: str, prefix: str) -> pd.DataFrame:
    if value_column not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[value_column])
    if valid.empty:
        return pd.DataFrame()

    pivot = valid.pivot_table(
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
