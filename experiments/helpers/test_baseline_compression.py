import gc
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from skimage import io as skio

from fused_ssim import fused_ssim3d


DATA_ROOT = Path("/scratch/papi/renner_data/r2_gaussian/data")
RESULTS_ROOT = Path("/scratch/papi/renner_data/r2_gaussian/models/vol_compress")
TEMP_JPEG_DIR = Path(RESULTS_ROOT / "temp")
REPORT_CSV = Path(RESULTS_ROOT / "baseline_compression.csv")
FLOAT_BYTES = 4

DATA_SOURCES = {
    "real": [
        DATA_ROOT / "real_dataset" / "cone_ntrain_25_angle_360",
    ],
    "synthetic": [
        DATA_ROOT / "synthetic_dataset" / "cone_ntrain_25_angle_360",
    ],
}
CompressionMetrics = Dict[str, Any]

def quantize_volume(volume: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Normalize and quantize a float32 volume into uint8."""
    vol_min = float(volume.min())
    vol_max = float(volume.max())
    rng = vol_max - vol_min
    safe_range = rng if rng > 0 else 1.0
    normalized = np.clip((volume - vol_min) / safe_range, 0.0, 1.0)
    uint8_volume = np.rint(normalized * 255.0).astype(np.uint8)
    return uint8_volume, vol_min, vol_max, safe_range


def compute_uint8_metrics(
    original_volume: np.ndarray, uint8_volume: np.ndarray, vol_min: float, rng: float
) -> Tuple[int, int, float, float]:
    """Compute storage sizes and reconstruction metrics for uint8 volume."""
    reconstructed = uint8_volume.astype(np.float32) / 255.0
    reconstructed = reconstructed * rng + vol_min
    psnr_score, ssim_score = compute_psnr_and_ssim(original_volume, reconstructed)
    uint8_size = uint8_volume.nbytes + 2 * FLOAT_BYTES  # store min+max
    uint8_zip_size = compress_uint8_volume(uint8_volume) + 2 * FLOAT_BYTES
    del reconstructed
    return uint8_size, uint8_zip_size, psnr_score, ssim_score


def compress_uint8_volume(uint8_volume: np.ndarray) -> int:
    """Return the size of ZIP-compressed uint8 volume."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=-1
    ) as zipf:
        zipf.writestr("volume.bin", uint8_volume.tobytes())
    size = buffer.getbuffer().nbytes
    buffer.close()
    return size


def compute_psnr_and_ssim(
    original_volume: np.ndarray, reconstructed_volume: np.ndarray, max_val=1.0
) -> Tuple[float, float]:
    """Compute PSNR/SSIM between floating point reference and reconstruction."""
    diff = reconstructed_volume.astype(np.float64) - original_volume.astype(np.float64)
    mse = float(np.mean(np.square(diff), dtype=np.float64))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((max_val**2) / mse)

    ssim_score = float("nan")
    with torch.no_grad():
        gt = (
            torch.from_numpy(original_volume)
            .to(device=torch.device("cuda"), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        recon = (
            torch.from_numpy(reconstructed_volume)
            .to(device=torch.device("cuda") , dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ssim_score = float(fused_ssim3d(recon, gt).item())

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return psnr, ssim_score


def jpeg_compression_metrics(
    dataset_name: str,
    instance_name: str,
    original_volume: np.ndarray,
    vol_min: float,
    rng: float,
    uint8_volume: np.ndarray,
) -> Tuple[int, float, float]:
    """Compress each slice into JPEG files and compute size + metrics."""
    instance_dir = TEMP_JPEG_DIR / dataset_name / instance_name
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)

    jpeg_paths: List[Path] = []
    for idx in range(uint8_volume.shape[0]):
        slice_arr = uint8_volume[idx]
        path = instance_dir / f"slice_{idx:04d}.jpg"
        skio.imsave(path, slice_arr, quality=75, check_contrast=False)
        jpeg_paths.append(path)

    jpeg_total = sum(path.stat().st_size for path in jpeg_paths)
    metadata_bytes = 2 * FLOAT_BYTES  # store min+max
    total_size = jpeg_total + metadata_bytes

    recon_volume = np.empty_like(original_volume)
    cached_slices: List[np.ndarray] = []
    for path in jpeg_paths:
        img = skio.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        cached_slices.append(img.astype(np.uint8))

    for slice_idx, cached in enumerate(cached_slices):
        recon_slice = cached.astype(np.float32) / 255.0
        recon_volume[slice_idx] = recon_slice * rng + vol_min

    psnr_score, ssim_score = compute_psnr_and_ssim(original_volume, recon_volume)

    shutil.rmtree(instance_dir)
    del recon_volume, cached_slices
    return total_size, psnr_score, ssim_score


def process_instance(dataset_name: str, instance_dir: Path) -> Optional[CompressionMetrics]:
    """Compute compression metrics for a single reconstructed volume."""
    vol_path = instance_dir / "vol_gt.npy"
    if not vol_path.exists():
        print(f"[WARN] Missing volume file: {vol_path}")
        return None

    print(f"[INFO] Processing {dataset_name}/{instance_dir.name}")
    volume = np.load(vol_path)
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)

    orig_size = vol_path.stat().st_size
    uint8_volume, vol_min, _, rng = quantize_volume(volume)

    uint8_size, uint8_zip_size, uint8_psnr, uint8_ssim = compute_uint8_metrics(
        volume, uint8_volume, vol_min, rng
    )

    jpeg_size, jpeg_psnr, jpeg_ssim = jpeg_compression_metrics(
        dataset_name,
        instance_dir.name,
        volume,
        vol_min,
        rng,
        uint8_volume,
    )

    metrics: CompressionMetrics = {
        "dataset": dataset_name,
        "instance": instance_dir.name,
        "original_size": orig_size,
        "uint8_size": uint8_size,
        "uint8_zip_size": uint8_zip_size,
        "uint8_psnr": uint8_psnr,
        "uint8_ssim": uint8_ssim,
        "jpeg_size": jpeg_size,
        "jpeg_psnr": jpeg_psnr,
        "jpeg_ssim": jpeg_ssim,
    }

    del volume, uint8_volume
    gc.collect()
    return metrics


def main() -> None:
    """Run the baseline compression sweep for every dataset in ``DATA_SOURCES``."""
    TEMP_JPEG_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: List[CompressionMetrics] = []
    for dataset_name, candidates in DATA_SOURCES.items():
        # Choose the first candidate path that exists for this dataset.
        dataset_dir = next((candidate for candidate in candidates if candidate.exists()), None)
        if dataset_dir is None:
            print(f"[WARN] No directory found for '{dataset_name}' dataset.")
            continue

        # Walk through each volume instance within the resolved dataset directory.
        for instance_dir in sorted(dataset_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            metrics = process_instance(dataset_name, instance_dir)
            if metrics is not None:
                all_metrics.append(metrics)

    if not all_metrics:
        print("[WARN] No metrics to report.")
        return

    # Summarize collected metrics in a pandas DataFrame for printing to stdout.
    df = pd.DataFrame(all_metrics)
    print(df.to_string(index=False))

    # Persist the same table to disk for downstream analysis.
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_CSV, index=False)
    print(f"[INFO] CSV report written to {REPORT_CSV}")


if __name__ == "__main__":
    main()
