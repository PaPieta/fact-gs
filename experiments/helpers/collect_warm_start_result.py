#!/usr/bin/env python3

"""Summarize warm-start experiment metrics across chest CT scans."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recon-grad-root",
        type=Path,
        required=True,
        help="Directory containing gradient-init recon outputs.",
    )
    parser.add_argument(
        "--recon-prior-root",
        type=Path,
        required=True,
        help="Directory containing prior-init recon outputs.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=20000,
        help="Total reconstruction steps (used for percentage calculations).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Comma-separated list of case names to process.",
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
    cases: Sequence[str] = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = []
    for case in cases:
        record = collect_case_metrics(
            case,
            args.recon_grad_root,
            args.recon_prior_root,
            args.total_steps,
        )
        if record:
            rows.append(record)

    if not rows:
        raise SystemExit("No warm-start metrics collected.")

    df = pd.DataFrame(rows)
    print(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("\nColumn averages:")
    print(df.mean(numeric_only=True).to_string())
    print(f"\nSaved warm-start metrics to {args.output_csv}")


def collect_case_metrics(
    case_name: str,
    recon_grad_root: Path,
    recon_prior_root: Path,
    total_steps: int,
) -> Optional[Dict[str, Optional[float]]]:
    grad_dir = recon_grad_root / case_name
    prior_dir = recon_prior_root / case_name

    if not grad_dir.exists() or not prior_dir.exists():
        print(f"Skipping {case_name}: missing directories.")
        return None

    grad_series = read_eval_series(grad_dir / "eval")
    prior_series = read_eval_series(prior_dir / "eval")

    if not grad_series.ssim or not prior_series.ssim:
        print(f"Skipping {case_name}: incomplete eval data.")
        return None

    max_ssim_recon = max(grad_series.ssim)
    max_psnr_recon = max(grad_series.psnr)
    max_ssim_prior = max(prior_series.ssim)
    max_psnr_prior = max(prior_series.psnr)

    iter_ssim_prior = find_iteration_threshold(
        prior_series.iterations, prior_series.ssim, max_ssim_recon
    )
    iter_psnr_prior = find_iteration_threshold(
        prior_series.iterations, prior_series.psnr, max_psnr_recon
    )

    return {
        "name": case_name,
        "max_ssim_recon": max_ssim_recon,
        "max_psnr_recon": max_psnr_recon,
        "max_ssim_recon_prior": max_ssim_prior,
        "max_psnr_recon_prior": max_psnr_prior,
        "iteration_percent_ssim": iteration_to_percent(iter_ssim_prior, total_steps),
        "iteration_percent_psnr": iteration_to_percent(iter_psnr_prior, total_steps),
    }


def iteration_to_percent(iteration: Optional[int], total_steps: int) -> Optional[float]:
    if iteration is None or total_steps <= 0:
        return None
    return (iteration / total_steps) * 100.0


class EvalSeries:
    __slots__ = ("iterations", "ssim", "psnr")

    def __init__(self, iterations: List[int], ssim: List[float], psnr: List[float]):
        self.iterations = iterations
        self.ssim = ssim
        self.psnr = psnr


def read_eval_series(eval_root: Path) -> EvalSeries:
    iterations: List[int] = []
    ssims: List[float] = []
    psnrs: List[float] = []

    if not eval_root.exists():
        return EvalSeries(iterations, ssims, psnrs)

    for eval_dir in sorted(
        (d for d in eval_root.iterdir() if d.is_dir()), key=lambda p: p.name
    ):
        iteration = parse_iteration(eval_dir.name)
        eval_file = eval_dir / "eval3d.yml"
        if not eval_file.exists():
            continue
        with open(eval_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        iterations.append(iteration)
        ssims.append(float(data.get("ssim_3d", math.nan)))
        psnrs.append(float(data.get("psnr_3d", math.nan)))

    return EvalSeries(iterations, ssims, psnrs)


def parse_iteration(dirname: str) -> int:
    if dirname.startswith("init"):
        return 0
    parts = dirname.split("_")
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
        try:
            return int(token.lstrip("0") or "0")
        except ValueError:
            continue
    # If no digits are found, fall back to 0
    return 0


def find_iteration_threshold(
    iterations: Sequence[int],
    values: Sequence[float],
    target: float,
) -> Optional[int]:
    if math.isnan(target):
        return None
    for iteration, value in zip(iterations, values):
        if not math.isnan(value) and value >= target:
            return iteration
    return None


if __name__ == "__main__":
    main()
