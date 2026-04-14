import hydra
import os
import numpy as np
import tigre ## For some reason, tigre needs to be imported before torch to avoid GPU errors
import torch
import sys

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel, initialize_gaussian
from fact_gs.r2_gaussian.utils.general_utils import safe_state
from fact_gs.r2_gaussian.dataset import SceneRecon, SceneVol
from fact_gs.r2_gaussian.utils.image_utils import metric_vol, metric_proj

from fact_gs import rasterize_proj, voxelize_vol


@hydra.main(config_path="config", config_name="default_test.yaml", version_base=None)
def test_model(config):
    """Hydra entry point for testing an existing model."""
    # Check if the model path exists
    if not os.path.exists(config.model.model_path):
        raise ValueError("Model path does not exist.")
        
    model_args = config.model

    print(f"Data source path: {model_args.data_source_path}")
    print(f"Model save path: {model_args.model_path}")

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)  

    gaussians = GaussianModel(None)
    initialize_gaussian(gaussians, model_args, -1)

    if model_args.target == "recon":
        print("Testing reconstruction model...")
        scene = SceneRecon(model_args, shuffle=False)
        evaluate_recon(gaussians, scene, model_args)

    elif model_args.target == "vol":
        print(f"Testing volume fitting model, comparing to {model_args.vol_name}...")
        scene = SceneVol(model_args, shuffle=False, file_name=model_args.vol_name)

    evaluate_volume(gaussians, scene, model_args)

def evaluate_volume(gaussians, scene, model_args):
    # Implementation for evaluating volume fitting
    
    voxelize_pkg = voxelize_vol(
        gaussians,
        scene.scanner_cfg["offOrigin"],
        scene.scanner_cfg["nVoxel"],
        scene.scanner_cfg["sVoxel"],
    )
    vol_pred = voxelize_pkg["vol"]
    vol_gt = scene.vol_gt

    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
    print(f"Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}")

def evaluate_recon(gaussians, scene, model_args):
    # Implementation for evaluating reconstruction
    validation_configs = [
        {"name": "render_train", "cameras": scene.getTrainCameras()},
        {"name": "render_test", "cameras": scene.getTestCameras()},
    ]
    psnr_2d, ssim_2d = None, None
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            images = []
            gt_images = []
            image_show_2d = []
            # Render projections
            show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
            for idx, viewpoint in enumerate(config["cameras"]):
                image = rasterize_proj(
                    viewpoint,
                    gaussians,
                )["render"]
                gt_image = viewpoint.original_image.to("cuda")
                images.append(image)
                gt_images.append(gt_image)
            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
            psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
            ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
            
            print(f"Evaluating {config['name']}: psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}")

if __name__ == "__main__":
    test_model()

