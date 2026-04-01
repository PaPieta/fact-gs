import hydra
import os
import os.path as osp
from random import randint
from tqdm import tqdm
import numpy as np
from fused_ssim import fused_ssim
import tigre ## For some reason, tigre needs to be imported before torch to avoid GPU errors
import torch
import sys
import yaml

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel, initialize_gaussian_from_proj, initialize_gaussian, initialize_gaussian_from_prior
from fact_gs.r2_gaussian.utils.general_utils import safe_state
from fact_gs.r2_gaussian.dataset import SceneRecon
from fact_gs.r2_gaussian.utils.loss_utils import l1_loss, tv_3d_loss
from fact_gs.r2_gaussian.utils.image_utils import metric_vol, metric_proj

from fact_gs import rasterize_proj, voxelize_vol
from fact_gs.utils.profile import setup_profiler
from fact_gs.utils.vol_utils import save_volume, visualize_gaussian_footprint, visualize_gaussian_position, save_error_maps



@hydra.main(config_path="config", config_name="default_recon.yaml", version_base=None)
def train_recon(config):
    """Hydra entry point for reconstruction training."""
    os.makedirs(config.model.model_path, exist_ok=True)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)  

    profiler = setup_profiler(config.profile, config.model.model_path)

    if profiler is not None:
        with profiler:
            optimize(config, profiler)

    else:
        optimize(config)


def optimize(config, profiler=None):
    """Run the full reconstruction optimization loop.

    Args:
        config: Hydra config with ``model``, ``optim`` and ``eval`` sections.
        profiler: Optional torch profiler context returned by ``setup_profiler``.
    """
    model_args, optim_args, eval_args = config.model, config.optim, config.eval
    scene = SceneRecon(model_args, shuffle=False)

    print(f"Data source path: {model_args.data_source_path}")
    print(f"Model save path: {scene.model_path}")

    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if model_args.scale_min > 0 and model_args.scale_max > 0:
        scale_bound = np.array([model_args.scale_min, model_args.scale_max]) * volume_to_world
    voxelizefunc = lambda x: voxelize_vol(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
    )

    # Gaussian densification params
    max_scale = optim_args.max_scale * volume_to_world if optim_args.max_scale else None
    densify_scale_threshold = (
        optim_args.densify_scale_threshold * volume_to_world
        if optim_args.densify_scale_threshold
        else None
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    if model_args.init_mode == "precomputed":
        initialize_gaussian(gaussians, model_args, None)
    elif model_args.init_mode in ["gradient", "intensity"]:
        initialize_gaussian_from_proj(gaussians, model_args, optim_args, scene)
    elif model_args.init_mode == "prior":
        initialize_gaussian_from_prior(gaussians, model_args)
    else:
        raise ValueError(f"Unknown initialization mode {model_args.init_mode}")
    scene.gaussians = gaussians
    gaussians.training_setup(optim_args)

    # Translate the configured percentage to an absolute cap for densification
    max_num_gaussians_limit = None
    if optim_args.max_num_gaussians is not None:
        initial_gaussians = getattr(model_args, "num_gaussians", None)
        if initial_gaussians is None or initial_gaussians <= 0:
            initial_gaussians = gaussians.get_xyz.shape[0]
        max_num_gaussians_limit = int(initial_gaussians * optim_args.max_num_gaussians)

    # Set up loss
    use_tv = optim_args.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = optim_args.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
        reduction_value = 3*(tv_vol_size-1)*tv_vol_size**2

    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, optim_args.steps), desc="Train", leave=False)
    for step in range(optim_args.steps + 1):
        step_start.record()
        # Update learning rate
        gaussians.update_learning_rate(step)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = rasterize_proj(viewpoint_cam, gaussians)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if optim_args.lambda_dssim > 0:
            loss_dssim = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + optim_args.lambda_dssim * loss_dssim

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            
            vol_pred = voxelize_vol(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction_value=reduction_value)
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + optim_args.lambda_tv * loss_tv
        loss["total"].backward()

        step_end.record()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            # Optional densification
            if optim_args.densify_gaussians:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if step < int(optim_args.densify_until_step_percent * optim_args.steps):
                    if (
                        step > optim_args.densify_from_step
                        and step % optim_args.densification_interval == 0
                    ):
                        gaussians.densify_and_prune(
                            optim_args.densify_grad_threshold,
                            optim_args.density_min_threshold,
                            optim_args.max_screen_size,
                            max_scale,
                            max_num_gaussians_limit,
                            densify_scale_threshold,
                            bbox,
                        )
                        # print(f"Number of Gaussians after densification: {gaussians.get_density.shape[0]}")
                        # print(f"Scale after densification, max: {gaussians.get_scaling.max().item():.4f}, mean: {gaussians.get_scaling.mean().item():.4f}, min: {gaussians.get_scaling.min().item():.4f}")
                        # print(f"Density after densification, max: {gaussians.get_density.max().item():.4e}, mean: {gaussians.get_density.mean().item():.4e}, min: {gaussians.get_density.min().item():.4e}")   
                

            # Optimization
            if step < optim_args.steps:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if step == optim_args.steps:
                tqdm.write(f"[STEP {step}] Saving Gaussians")
                scene.save(step, voxelizefunc, vol_format="tiff")

            # Progress bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if step == optim_args.steps:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            log_training_status(
                step,
                metrics,
                step_start.elapsed_time(step_end),
                optim_args.steps,
                eval_args,
                scene,
                lambda x, y: rasterize_proj(x, y),
                voxelizefunc,
                model_args.init_mode,
            )

            if profiler is not None:
                profiler.step()
    torch.cuda.empty_cache()

def log_training_status(
    step,
    metrics_train,
    elapsed,
    max_steps,
    eval_args,
    scene: SceneRecon,
    renderFunc,
    voxelizeFunc,
    init_mode,
):
    """Evaluate/visualize model checkpoints and persist metrics.

    Args:
        step: Current global training step.
        metrics_train: Dictionary with scalar loss/learning-rate values.
        elapsed: CUDA event duration reported in milliseconds.
        max_steps: Final step count that signals completion.
        eval_args: Evaluation sub-config that controls cadence and visualization.
        scene: Scene wrapper that stores Gaussians, volumes and metadata.
        renderFunc: Callable that renders projections given a camera and model.
        voxelizeFunc: Callable that voxelizes the global Gaussian model.
        init_mode: String describing how Gaussians were initialized (for logging).
    """

    if (eval_args.eval_in_training and step % eval_args.every_n_steps == 0 and step != 0) or (eval_args.eval_end and step == max_steps) or (eval_args.eval_start and step == 0):
        # Evaluate 2D rendering performance
        if step == 0:
            eval_save_path = osp.join(scene.model_path, "eval", f"init_{init_mode}")
        else:
            eval_save_path = osp.join(scene.model_path, "eval", f"step_{step:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

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
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

        # Evaluate 3D reconstruction performance
        voxelize_pkg = voxelizeFunc(scene.gaussians)
        vol_pred = voxelize_pkg["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d.item(),
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        
        tqdm.write(
            f"[STEP {step}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        if eval_args.visualize_at_eval:
            vol_save_path = osp.join(eval_save_path, "vol_pred.tiff")
            save_volume(vol_pred, vol_save_path, save_preview=True, save_volume=True)
            if step == 0:
                vol_gt_save_path = osp.join(scene.model_path, "vol_gt.tiff")
                save_volume(vol_gt, vol_gt_save_path, save_preview=True, save_volume=True)
        
        if eval_args.visualize_gaussians:
            gauss_position_path = osp.join(eval_save_path, "gaussian_positions/")
            gauss_footprint_path = osp.join(eval_save_path, "gaussian_footprints/")
            error_maps_path = osp.join(eval_save_path, "error_maps/")
            os.makedirs(gauss_position_path, exist_ok=True)
            os.makedirs(gauss_footprint_path, exist_ok=True)
            os.makedirs(error_maps_path, exist_ok=True)

            visualize_gaussian_position(gauss_position_path, 
                                        scene.gaussians.get_xyz,
                                        voxelize_pkg["radii"], 
                                        scene.scanner_cfg["sVoxel"],
                                        scene.scanner_cfg["dVoxel"],
                                        scene.scanner_cfg["offOrigin"],)
            visualize_gaussian_footprint(gauss_footprint_path, 
                                        scene.gaussians.get_xyz, 
                                        voxelize_pkg["radii"], 
                                        voxelize_pkg["conics"],
                                        scene.gaussians.get_density, 
                                        scene.scanner_cfg["sVoxel"],
                                        scene.scanner_cfg["dVoxel"],    
                                        scene.scanner_cfg["offOrigin"],)
            save_error_maps(error_maps_path, vol_pred, vol_gt)

        if step == max_steps:
            # Save PSNR and SSIM metrics to a yaml file
            eval_dict = {
                "psnr_3d": psnr_3d,
                "ssim_3d": ssim_3d.item(),
            }
            # Take a step back from model path
            eval_save_path = osp.dirname(scene.model_path)
            yaml_name = f"{osp.basename(scene.model_path)}_metrics_final.yml"
            with open(osp.join(eval_save_path, yaml_name), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    train_recon()
    
