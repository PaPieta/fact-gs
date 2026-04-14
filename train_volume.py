import hydra
import os
import os.path as osp
import time
from tqdm import tqdm
import numpy as np

from fused_ssim import fused_ssim3d
import tigre ## For some reason, tigre needs to be imported before torch to avoid GPU errors
import torch
import sys
import yaml

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel, initialize_gaussian_from_vol
from fact_gs.r2_gaussian.utils.general_utils import safe_state
from fact_gs.r2_gaussian.dataset import SceneVol
from fact_gs.r2_gaussian.utils.loss_utils import l1_loss, tv_3d_loss
from fact_gs.r2_gaussian.utils.image_utils import metric_vol

from fact_gs import voxelize_vol
from fact_gs.utils.profile import setup_profiler
from fact_gs.utils.quantization_utils import quantize_gaussians, report_model_size
from fact_gs.utils.vol_utils import save_volume, visualize_gaussian_footprint, visualize_gaussian_position, save_error_maps


@hydra.main(version_base=None, config_path="config", config_name="default_volume.yaml")
def train_volume(config):
    """Hydra entry point for volume-only optimization."""
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
    """Optimize a Gaussian model directly against a target volume.

    Args:
        config: Hydra config with ``model``, ``optim`` and ``eval`` sections.
        profiler: Optional torch profiler context manager.
    """
    model_args, optim_args, eval_args = config.model, config.optim, config.eval
    scene = SceneVol(model_args, file_name=model_args.vol_name)
    training_time_seconds = 0.0
    eval_and_save_time_seconds = 0.0

    print(f"Data source path: {model_args.data_source_path}")
    print(f"Volume name: {model_args.vol_name}")
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
        raise NotImplementedError("Precomputed initialization not implemented for volume compression.")
    elif model_args.init_mode in ["gradient", "intensity"]:
        initialize_gaussian_from_vol(gaussians, model_args, optim_args, scene)
    else:
        raise ValueError(f"Unknown initialization mode {model_args.init_mode}")
    scene.gaussians = gaussians
    gaussians.training_setup(optim_args)

    # Translate the configured percentage to an absolute cap for densification
    max_num_gaussians_limit = int(model_args.num_gaussians * optim_args.max_num_gaussians)

    if optim_args.quantize:
        print("Quantization enabled.")
    report_model_size(gaussians, optim_args)

    # Set up loss
    use_tv = optim_args.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = scanner_cfg["nVoxel"]
        reduction_value = 3*(tv_vol_size[0]-1)*tv_vol_size[1]*tv_vol_size[2] + \
                          3*tv_vol_size[0]*(tv_vol_size[1]-1)*tv_vol_size[2] + \
                          3*tv_vol_size[0]*tv_vol_size[1]*(tv_vol_size[2]-1)

    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, optim_args.steps), desc="Train", leave=False)
    latest_eval_metrics = None
    for step in range(optim_args.steps + 1):
        step_time_start = time.perf_counter()
        # Update learning rate
        gaussians.update_learning_rate(step)

        if optim_args.quantize:
            quantize_gaussians(gaussians, optim_args)

        vol_pkg = voxelize_vol(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            get_pos_radii_buffer_for_grad=True,
        )
        vol, visibility_filter, volspace_points, radii = (
            vol_pkg["vol"],
            vol_pkg["visibility_filter"],
            vol_pkg["volspace_points"],
            vol_pkg["radii"],
        )
        
        gt_vol = scene.vol_gt.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(vol, gt_vol)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if optim_args.lambda_dssim > 0:
            loss_dssim = 1.0 - fused_ssim3d(vol.unsqueeze(0).unsqueeze(0), gt_vol.unsqueeze(0).unsqueeze(0))
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + optim_args.lambda_dssim * loss_dssim

        if use_tv:
            loss_tv = tv_3d_loss(vol, reduction_value=reduction_value)
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + optim_args.lambda_tv * loss_tv
        loss["total"].backward()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            
            # Optional densification
            if optim_args.densify_gaussians:
                gaussians.add_densification_stats_3d(volspace_points, visibility_filter)
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

            training_time_seconds += time.perf_counter() - step_time_start
            step_non_training_time_start = time.perf_counter()

            # Save gaussians
            if step == optim_args.steps:
                tqdm.write(f"[STEP/ITER {step}] Saving Gaussians")
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
            eval_metrics = log_training_status(
                step,
                metrics,
                training_time_seconds,
                eval_and_save_time_seconds,
                optim_args.steps,
                eval_args,
                scene,
                voxelizefunc,
                model_args.init_mode,
            )

            if eval_metrics is not None:
                latest_eval_metrics = eval_metrics

            eval_and_save_time_seconds += time.perf_counter() - step_non_training_time_start

            # Early stopping based on 3D SSIM
            if optim_args.ssim3d_early_stop and eval_metrics is not None:
                threshold = getattr(optim_args, "ssim3d_early_stop_threshold", 0.0)
                ssim_value = eval_metrics.get("ssim_3d")
                if ssim_value is not None and threshold > 0 and ssim_value >= threshold:
                    iteration_str = f"{eval_metrics['iteration']:.2f}"
                    tqdm.write(
                        f"[STEP {step}] Early stopping triggered at iteration {iteration_str} "
                        f"(SSIM={eval_metrics['ssim_3d']:.4f} >= {threshold:.3f})"
                    )
                    if step != optim_args.steps:
                        scene.save(step, voxelizefunc, vol_format="tiff")
                    save_final_metrics(scene.model_path, eval_metrics, training_time_seconds)
                    break

            # Early stopping based on accumulated training time
            time_limit_seconds = getattr(optim_args, "training_time_limit_seconds", 0.0)
            if time_limit_seconds > 0 and training_time_seconds >= time_limit_seconds:
                tqdm.write(
                    f"[STEP {step}] Stopping because training time "
                    f"{training_time_seconds:.2f}s exceeded limit {time_limit_seconds:.2f}s"
                )
                if step != optim_args.steps:
                    scene.save(step, voxelizefunc, vol_format="tiff")
                if latest_eval_metrics is not None:
                    save_final_metrics(scene.model_path, latest_eval_metrics, training_time_seconds)
                break

            if profiler is not None:
                profiler.step()
    progress_bar.close()
    torch.cuda.empty_cache()

def save_final_metrics(model_path, metrics, training_time_seconds):
    """Save final evaluation metrics to a YAML file for easy retrieval."""
    if not metrics:
        return
    eval_save_path = osp.dirname(model_path)
    yaml_name = f"{osp.basename(model_path)}_metrics_final.yml"
    payload = {
        "psnr_3d": metrics.get("psnr_3d"),
        "ssim_3d": metrics.get("ssim_3d"),
        "time_training_seconds": training_time_seconds,
    }
    with open(osp.join(eval_save_path, yaml_name), "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

def log_training_status(
    step,
    metrics_train,
    training_time_seconds,
    eval_and_save_time_seconds,
    max_steps,
    eval_args,
    scene: SceneVol,
    voxelizeFunc,
    init_mode,
    force_eval=False,
):
    """Evaluate/visualize volume training progress at the requested cadence.

    Args:
        step: Current global training step.
        metrics_train: Dictionary with scalar loss/learning-rate values.
        training_time_seconds: Total time spent on training.
        eval_and_save_time_seconds: Total time spent on evaluation and saving.
        max_steps: Maximum number of training steps.
        eval_args: Evaluation sub-config that controls cadence and visualization.
        scene: Scene wrapper that exposes volumes and Gaussian parameters.
        voxelizeFunc: Callable that converts current Gaussians into a volume.
        init_mode: String describing how Gaussians were initialized (for logging).
    """

    iter_num = step  # For volume optimization, each step corresponds to one iteration.

    eval_metrics = None

    should_eval = force_eval or (
        (eval_args.eval_in_training and step % eval_args.every_n_steps == 0 and step != 0)
        or (eval_args.eval_end and step == max_steps)
        or (eval_args.eval_start and step == 0)
        or (eval_args.extra_eval_iter_num is not None and iter_num == eval_args.extra_eval_iter_num)
    )

    if should_eval:
        # Evaluate 2D rendering performance
        if step == 0:
            eval_save_path = osp.join(scene.model_path, "eval", f"init_{init_mode}")
        else:
            eval_save_path = osp.join(scene.model_path, "eval", f"step_{step:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        time_dict = {
            "training_time_seconds": float(training_time_seconds),
            "eval_and_save_time_seconds": float(eval_and_save_time_seconds),
        }
        with open(osp.join(eval_save_path, "time.yml"), "w") as f:
            yaml.dump(time_dict, f, default_flow_style=False, sort_keys=False)
        
        # Evaluate 3D reconstruction performance
        voxelize_pkg = voxelizeFunc(scene.gaussians)
        vol_pred = voxelize_pkg["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
        eval_metrics = {
            "psnr_3d": float(psnr_3d),
            "ssim_3d": float(ssim_3d.item()),
            "step": float(step),
            "iteration": float(iter_num),
        }
        eval_dict = {
            "psnr_3d": eval_metrics["psnr_3d"],
            "ssim_3d": eval_metrics["ssim_3d"],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        tqdm.write(
            f"[STEP/ITER {step}] Training Time: {training_time_seconds:.2f}s. Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}"
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

        if eval_args.extra_eval_iter_num is not None and iter_num == eval_args.extra_eval_iter_num:
            eval_dict = {
                "psnr_3d": eval_metrics["psnr_3d"],
                "ssim_3d": eval_metrics["ssim_3d"],
                "training_time_seconds": training_time_seconds,
                "eval_and_save_time_seconds": eval_and_save_time_seconds,
            }
            eval_save_path = osp.dirname(scene.model_path)
            yaml_name = f"{osp.basename(scene.model_path)}_metrics_{int(iter_num)}.yml"
            with open(osp.join(eval_save_path, yaml_name), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        if step == max_steps:
            save_final_metrics(scene.model_path, eval_metrics, training_time_seconds)

    return eval_metrics

if __name__ == "__main__":
    train_volume()
