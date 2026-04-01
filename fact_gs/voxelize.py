import sys
import torch

from gs_voxelizer import voxelize, optim_to_render 

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel

def _as_tuple(value, cast_fn=float):
    """Convert scalars/tensors/arrays to tuples of ``cast_fn`` outputs.

    Args:
        value: Scalar or sequence that represents per-axis sizes.
        cast_fn: Callable applied to every entry of ``value``.

    Returns:
        Tuple of processed values suitable for CUDA kernels.
    """
    # Accept torch tensors / numpy arrays / lists and turn them into Python tuples
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(cast_fn(v) for v in value)
    return (cast_fn(value),)

def voxelize_vol(
    pc: GaussianModel,
    vol_center_pos,
    vol_size_voxel,
    vol_size_world,
    density=None,
    dim_order="zyx",
    get_pos_radii_buffer_for_grad=False,
):
    """Rasterize Gaussians into a dense 3D voxel grid.

    Args:
        pc: Gaussian model whose tensors encode point attributes.
        vol_center_pos: Volume center in world coordinates.
        vol_size_voxel: Volume size in voxels (x, y, z).
        vol_size_world: Physical volume dimensions in world units.
        density: Optional per-Gaussian density tensor override.
        dim_order: Output tensor layout, either ``zyx`` or ``xyz``.
        get_pos_radii_buffer_for_grad: When ``True`` retains buffers for grads.

    Returns:
        Dictionary storing the predicted volume, conic parameters and masks.
    """
    # Convert to tuple and flip to ZYX
    vol_size_voxel_tuple = _as_tuple(vol_size_voxel, int)[::-1]
    vol_size_world_tuple = _as_tuple(vol_size_world, float)[::-1]
    vol_center_pos_tuple = _as_tuple(vol_center_pos, float)[::-1]

    if get_pos_radii_buffer_for_grad:
        pos_radii_buffer = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda").contiguous()
    else:
        pos_radii_buffer = None

    pos3d_viz_radii, conics, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pc.get_xyz,
        pc.get_scaling,
        pc.get_rotation,
        pc.get_density,
        vol_size_voxel_tuple,
        vol_size_world_tuple,
        vol_center_pos_tuple,
        pos_radii_buffer,
    )
    if density is None:
        density = pc.get_density

    # volspace_points = pos3d_viz_radii[:, :3]
    if pos3d_viz_radii.requires_grad:
        pos3d_viz_radii.retain_grad()

    # Voxelize gaussians
    vol_pred = voxelize.voxelize_gaussians(
        pos3d_viz_radii,
        conics,
        density,
        vol_size_voxel_tuple,
        tile_min,
        tile_max,
        num_tiles_hit,
        use_per_gaussian_backward=True,
    )
    if dim_order == "zyx":
        vol_pred = torch.permute(vol_pred, (2, 1, 0, 3))
    elif dim_order == "xyz":
        pass
    else:
        raise ValueError("Unsupported dimension order!")

    return {
        "vol": vol_pred.squeeze(),
        "visibility_filter": pos3d_viz_radii[:, 3] > 0,
        "volspace_points": pos3d_viz_radii,
        "radii": pos3d_viz_radii[:, 3],
        "conics": conics,
    }
