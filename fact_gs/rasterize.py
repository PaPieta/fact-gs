import sys
import torch
import math

from gs_ct_rasterizer import optim_to_render, rasterize

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel
from fact_gs.r2_gaussian.dataset.cameras import Camera


def rasterize_proj(
    viewpoint_camera: Camera,
    pc: GaussianModel,
):
    """Render a single projection image from the Gaussian model.

    Args:
        viewpoint_camera: Camera definition that stores transforms, FoV and mode.
        pc: Gaussian model containing learnable position/orientation/density tensors.

    Returns:
        Dictionary with rendered image, intermediate buffers and visibility flags.
    """
    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    pos2d, conics_mu, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pc.get_xyz,
        pc.get_scaling,
        pc.get_rotation,
        pc.get_density,
        viewpoint_camera.world_view_transform,
        viewpoint_camera.full_proj_transform,
        tanfovx,
        tanfovy,
        viewpoint_camera.image_height,
        viewpoint_camera.image_width,
        viewpoint_camera.mode,
        pos2d_buffer=torch.zeros_like(pc.get_xyz[:, :2], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda").contiguous(),
    )

    if pos2d.requires_grad:
        pos2d.retain_grad()

    rendered_image = rasterize.rasterize_gaussians(
        pos2d,
        conics_mu,
        pc.get_density,
        tile_min,
        tile_max,
        num_tiles_hit,
        viewpoint_camera.image_height,
        viewpoint_camera.image_width,
        use_per_gaussian_backward=True,
    ).permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": pos2d,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
