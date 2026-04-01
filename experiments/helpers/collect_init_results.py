#!/usr/bin/env python3

"""Aggregate initialization experiment metrics into a CSV summary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import tifffile  # type: ignore

from fused_ssim import fused_ssim3d

# (dataset_folder_name, data_name)
DATASETS: Tuple[Tuple[str, str], ...] = (
    ("brain_cone", "brain"),
    ("pancreas1_cone", "pancreas"),
    ("walnut_1_cone", "walnut"),
    ("crick_1_cone", "cricket"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmstart-data-root",
        type=Path,
        required=True,
        help="Path to warm-start dataset directory.",
    )
    parser.add_argument(
        "--recon-save-root",
        type=Path,
        required=True,
        help="Directory containing reconstruction outputs.",
    )
    parser.add_argument(
        "--vol-save-root",
        type=Path,
        required=True,
        help="Directory containing volume fitting outputs.",
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
    rows = []
    for dataset_folder, dataset_name in DATASETS:
        print(f"Collecting metrics for {dataset_name}")
        try:
            row = collect_dataset_metrics(
                dataset_folder,
                dataset_name,
                args.warmstart_data_root,
                args.recon_save_root,
                args.vol_save_root,
            )
        except FileNotFoundError as exc:
            print(f"  Skipping {dataset_name}: {exc}")
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("No metrics collected. Check dataset paths.")

    df = pd.DataFrame(rows)
    print(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved metrics to {args.output_csv}")


def collect_dataset_metrics(
    dataset_folder: str,
    dataset_name: str,
    warmstart_root: Path,
    recon_root: Path,
    vol_root: Path,
) -> Dict[str, float]:
    dataset_dir = warmstart_root / dataset_folder
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    vol_gt = load_volume(dataset_dir, "vol_gt")
    vol_prior = load_volume(dataset_dir, "vol_prior")

    ssim_prior = compute_ssim(vol_prior, vol_gt)

    vol_prior_eval_dir = vol_root / dataset_name / "eval" / "step_000050"
    vol_pred_path = vol_prior_eval_dir / "vol_pred.tiff"
    if not vol_pred_path.exists():
        raise FileNotFoundError(f"Missing vol_pred: {vol_pred_path}")
    vol_pred = load_volume_from_path(vol_pred_path)
    ssim_prior_train = compute_ssim(vol_pred, vol_gt)

    recon_eval_dir = recon_root / dataset_name / "eval"
    ssim_fdk_grad = read_eval_ssim(recon_eval_dir / "init_gradient" / "eval3d.yml")
    ssim_fdk_int = read_eval_ssim(recon_eval_dir / "init_intensity" / "eval3d.yml")

    return {
        "data_name": dataset_name,
        "ssim_prior": ssim_prior,
        "ssim_fdk_int": ssim_fdk_int,
        "ssim_fdk_grad": ssim_fdk_grad,
        "ssim_prior_train": ssim_prior_train,
    }


def load_volume(dataset_dir: Path, volume_name: str) -> np.ndarray:
    for ext in (".npy", ".tiff", ".tif"):
        candidate = dataset_dir / f"{volume_name}{ext}"
        if candidate.exists():
            return load_volume_from_path(candidate)
    raise FileNotFoundError(f"Volume '{volume_name}' not found in {dataset_dir}")


def load_volume_from_path(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume path does not exist: {path}")

    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    if path.suffix.lower() in {".tiff", ".tif"}:
        volume = tifffile.imread(path)
        volume = volume.astype(np.float32) / 255.0
        return volume

    raise ValueError(f"Unsupported volume extension: {path.suffix}")


def compute_ssim(volume_a: np.ndarray, volume_b: np.ndarray) -> float:
    tensor_a = torch.from_numpy(volume_a).float().cuda()
    tensor_b = torch.from_numpy(volume_b).float().cuda()
    ssim_value = fused_ssim3d(tensor_a.unsqueeze(0).unsqueeze(0), tensor_b.unsqueeze(0).unsqueeze(0))
    if hasattr(ssim_value, "item"):
        return float(ssim_value.item())
    if isinstance(ssim_value, np.ndarray):
        return float(ssim_value.squeeze())
    if isinstance(ssim_value, torch.Tensor):
        return float(ssim_value.detach().cpu().item())
    return float(ssim_value)


def read_eval_ssim(yaml_path: Path) -> float:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing eval file: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    value = data.get("ssim_3d", math.nan)
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)


if __name__ == "__main__":
    main()
