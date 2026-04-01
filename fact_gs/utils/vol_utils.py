# Inspired by image-gs image_utils.py

import os
from typing import Tuple

import tifffile 
import numpy as np
import torch
from scipy.ndimage import sobel
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

GAUSSIAN_ZOOM = 2
PLOT_DPI = 100.0

ALLOWED_VOLUME_FORMATS = {".tiff", ".tif"}

def get_grid(x_dim, y_dim, z_dim, x_lim=np.asarray([0,1]), y_lim=np.asarray([0,1]), z_lim=np.asarray([0,1])):
    """Create a dense 3D grid centered inside each voxel cell.

    Args:
        x_dim/y_dim/z_dim: Number of grid points per axis.
        x_lim/y_lim/z_lim: Axis-aligned bounds for the grid sampling.

    Returns:
        Tensor of shape ``(x_dim, y_dim, z_dim, 3)`` with XYZ coordinates.
    """
    x = torch.linspace(x_lim[0], x_lim[1], x_dim + 1)[:-1] + 0.5 / x_dim
    y = torch.linspace(y_lim[0], y_lim[1], y_dim + 1)[:-1] + 0.5 / y_dim
    z = torch.linspace(z_lim[0], z_lim[1], z_dim + 1)[:-1] + 0.5 / z_dim
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    return grid

def compute_volume_gradients(image):
    """Compute Sobel gradients along X/Y/Z axes.

    Args:
        image: Volume tensor/array with shape ``(1, D, H, W)``.

    Returns:
        Tuple of ``(grad_x, grad_y, grad_z)`` arrays.
    """

    gz = sobel(image[0], 0)
    gy = sobel(image[0], 1)
    gx = sobel(image[0], 2)

    return gx, gy, gz


def normalize_volume(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """Normalize volumetric TIFF data to float32 in [0, 1].

    Args:
        data: Loaded TIFF volume.

    Returns:
        Normalized volume and the detected bit depth.
    """
    if data.dtype == np.uint8:
        return (data.astype(np.float32) / 255.0), 8
    if data.dtype == np.uint16:
        return (data.astype(np.float32) / 65535.0), 16
    if data.dtype == np.float16:
        return data.astype(np.float32), 16
    if data.dtype in (np.float32, np.float64):
        return data.astype(np.float32), 32
    raise ValueError(f"Unsupported TIFF dtype: {data.dtype}")

def ensure_numpy(volume):
    """Convert tensors to ``np.ndarray`` while detaching gradients.

    Args:
        volume: numpy array or tensor representing the volume.

    Returns:
        numpy.ndarray with all computations moved to CPU.
    """
    if isinstance(volume, np.ndarray):
        return volume
    if torch.is_tensor(volume):
        # if volume.requires_grad:
        #     volume = volume.detach()
        return volume.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(volume)}")

def save_volume(volume, path, save_preview=False, save_volume=True):
    """Save a normalized volume, optionally generating preview slices.

    Args:
        volume: Input volume as numpy array or tensor.
        path: Destination TIFF path.
        save_preview: When ``True`` export mid-plane JPG slices.
        save_volume: Toggle for saving the TIFF volume itself.
    """
    # Move to CPU if necessary
    data = np.squeeze(ensure_numpy(volume)).astype(np.float32)
    # Rescale to 0-1 range
    # data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    # Clip to [0, 1]
    data = np.clip(data, 0.0, 1.0)
    # Convert to uint8
    data = (data * 255.0).astype(np.uint8)
    if save_volume:
        tifffile.imwrite(path, data)
    if save_preview:
        # Create preview folder
        preview_dir = os.path.join(os.path.dirname(path), "previews")
        # Append file name without extension
        preview_dir = os.path.join(preview_dir, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(preview_dir, exist_ok=True)
        # Get mid slice in each axis and save as JPG

        center_loc = np.array(data.shape) // 2
        mid_slices = [
            data[center_loc[0], :, :],
            data[:, center_loc[1], :],
            data[:, :, center_loc[2]],
        ]
        dims = ["YZ", "XZ", "XY"]
        for i, slice_ in enumerate(mid_slices):
            preview_path = os.path.join(preview_dir, f"slice_id{center_loc[i]}_{dims[i]}.jpg")
            skimage.io.imsave(preview_path, slice_, check_contrast=False)

def get_psnr(vol1, vol2, max_value=1.0):
    """Compute PSNR between two volumes stored as tensors.

    Args:
        vol1: Ground-truth tensor.
        vol2: Reconstructed tensor.
        max_value: Intensity ceiling used in the PSNR calculation.

    Returns:
        Scalar PSNR value in decibels.
    """
    mse = torch.mean((vol1-vol2)**2)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr

def get_2d_axis_aligned_conics(conics, plane='XY'):
    """Extract 2D conic coefficients for a chosen slice.

    Args:
        conics: Tensor with per-Gaussian conic coefficients.
        plane: Slice orientation: ``XY``, ``XZ`` or ``YZ``.

    Returns:
        Tensor of shape ``(N, 3)`` with ``A, B, C`` coefficients.
    """
    # Warning, the order is flipped, similarly to how we flip xyz_filt (dim_remaining[2-i,[1,0]])
    if plane == 'XY':
        A = conics[:, 3]
        B = conics[:, 1]
        C = conics[:, 0]
    elif plane == 'XZ':
        A = conics[:, 5]
        B = conics[:, 2]
        C = conics[:, 0]
    elif plane == 'YZ':
        A = conics[:, 5]
        B = conics[:, 4]
        C = conics[:, 3]
    else:
        raise ValueError(f"Invalid plane: {plane}")
    conics_2d = torch.stack([A, B, C], dim=-1)
    return conics_2d

def visualize_gaussian_footprint(save_path, xyz, radii, conics, feat, dim_size, voxel_sizes, off_origin, alpha=0.8, select_percent=0.2):
    """Render the ellipse footprint of Gaussians on orthogonal slices.

    Args:
        save_path: Base path prefix for produced PNGs.
        xyz: Gaussian positions.
        radii: Effective radius per Gaussian.
        conics: Conic parameters returned by voxelization.
        feat: Feature/density tensor for ranking Gaussians.
        dim_size: Number of pixels in the plotting canvas per axis.
        voxel_sizes: Physical voxel size expressed per axis.
        off_origin: Origin offset describing the dataset bounding box.
        alpha: Opacity of plotted ellipses.
        select_percent: Fraction of strongest Gaussians to visualize.
    """
    xyz = xyz.detach().cpu().clone().numpy()
    radii = radii.detach().cpu().clone().numpy()
    conics = conics.detach().cpu().clone().numpy()
    feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

    dim_names = ["XY", "XZ", "YZ"]
    dim_remaining = np.array([[1,2], [0,2], [0,1]])

    for i in range(3):
        center_loc = off_origin[i]
        voxel_size = voxel_sizes[i]
        # Get indices of gaussians that intersect with the slice
        g_idx = np.where((np.abs(xyz[:, 2-i] - (center_loc + voxel_size / 2) ) <= radii*voxel_size))[0]
        if len(g_idx) == 0:
            print(f"No gaussians intersect with {dim_names[i]} slice at {center_loc}")
            continue
        # Filter gaussians
        xy_filt = xyz[g_idx, :][:, dim_remaining[2-i,[1,0]]]
        conics_filt = conics[g_idx]
        feat_filt = feat[g_idx]
        feat_filt = feat_filt/feat_filt.max()
        # Get 2D conics
        conics_filt_2D = get_2d_axis_aligned_conics(torch.from_numpy(conics_filt), plane=dim_names[i])
        # Eignedecoposition to get rotation angles
        vals, vecs = np.linalg.eig(np.array([ [conics_filt_2D[:,0], conics_filt_2D[:,1]],
                                             [conics_filt_2D[:,1], conics_filt_2D[:,2]] ]).transpose(2,0,1))
        # Order eigenvalues and eigenvectors
        order = np.argsort(vals, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        vecs = np.take_along_axis(vecs, order[:, np.newaxis, :], axis=2)
        major_vec = vecs[:, :, -1]        
        # vecs = np.take_along_axis(vecs, order[:,:,np.newaxis], axis=2).squeeze(2)

        # Filter by feature strength
        num_to_select = max(1, int(len(feat_filt) * select_percent))
        top_indices = np.argsort(feat_filt[:,0])[-num_to_select:]
        # Add 10% random avoiding duplicates
        num_random = max(1, int(len(feat_filt) * 0.1))
        random_indices = np.random.choice(len(feat_filt), num_random, replace=False)
        top_indices = np.unique(np.concatenate([top_indices, random_indices]))

        xy_filt = xy_filt[top_indices]
        vals = vals[top_indices]
        major_vec = major_vec[top_indices]
        feat_filt = feat_filt[top_indices]

        # Get ellipse parameters
        width =voxel_sizes[dim_remaining[i][0]]/np.sqrt(vals[:,0]) * GAUSSIAN_ZOOM 
        height=voxel_sizes[dim_remaining[i][1]]/np.sqrt(vals[:,1]) * GAUSSIAN_ZOOM
        angle = np.degrees(np.arctan2(major_vec[:,0], major_vec[:,1]))

        fig = plt.figure(figsize=(8,8))
        fig.set_dpi(PLOT_DPI)
        fig.patch.set_facecolor('black')
        fig.set_size_inches(w=dim_size[dim_remaining[i][0]]*2, h=dim_size[dim_remaining[i][1]]*2, forward=False)
        ax = plt.gca()
        ax.set_facecolor('black')
        for gid in range(0, len(xy_filt)):
            fc = (feat_filt[gid].item(), feat_filt[gid].item(), feat_filt[gid].item())
            shift_zero = -(off_origin[dim_remaining[i][1]] - dim_size[dim_remaining[i][1]] //2)
            y_pos = dim_size[dim_remaining[i][1]] - (xy_filt[gid, 1] + 2 * shift_zero) # Invert y-axis for plotting
            ellipse = Ellipse(xy=(xy_filt[gid, 0], y_pos), width=width[gid], height=height[gid],
                              angle=angle[gid], alpha=alpha, ec=None, fc=fc, lw=None)
            ax.add_patch(ellipse)
        plt.xlim(off_origin[dim_remaining[i][0]] - dim_size[dim_remaining[i][0]] / 2, 
                 off_origin[dim_remaining[i][0]] + dim_size[dim_remaining[i][0]] / 2)
        plt.ylim(off_origin[dim_remaining[i][1]] - dim_size[dim_remaining[i][1]] / 2,
                 off_origin[dim_remaining[i][1]] + dim_size[dim_remaining[i][1]] / 2)
        plt.axis('off')
        plt.tight_layout()
        suffix = f"slice_pos{center_loc}_{dim_names[i]}"
        plt.savefig(f"{save_path}{suffix}.png", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI,facecolor=fig.get_facecolor())
        plt.close()

def visualize_gaussian_position(save_path, xyz, radii, dim_size, voxel_sizes, off_origin, color="#7bf1a8", size=3, every_n=10, alpha=0.8):
    """Scatter Gaussian centers along orthogonal slices.

    Args:
        save_path: Base path prefix for PNG exports.
        xyz: Gaussian center tensor.
        radii: Effective radius per Gaussian.
        dim_size: Number of pixels in the plotting canvas per axis.
        voxel_sizes: Physical voxel size per axis.
        off_origin: Bounding box origin used to locate slices.
        color: Hex color for scatter points.
        size: Marker size in matplotlib points.
        every_n: Down-sampling factor to reduce clutter.
        alpha: Marker opacity.
    """
    xyz = xyz.detach().cpu().clone().numpy()
    radii = radii.detach().cpu().clone().numpy()

    dim_names = ["XY", "XZ", "YZ"]
    dim_remaining = np.array([[1,2], [0,2], [0,1]])

    for i in range(3):
        center_loc = off_origin[i]
        voxel_size = voxel_sizes[i]
        # Get indices of gaussians that intersect with the slice
        g_idx = np.where((np.abs(xyz[:, 2-i] - (center_loc + voxel_size / 2) ) <= radii*voxel_size))[0]
        if len(g_idx) == 0:
            print(f"No gaussians intersect with {dim_names[i]} slice at {center_loc}")
            continue
        # Filter gaussians
        xy_filt = xyz[g_idx, :][:, dim_remaining[2-i,[1,0]]]
        xy_filt = xy_filt[::every_n]

        fig = plt.figure(figsize=(8,8))
        fig.set_dpi(PLOT_DPI)
        fig.patch.set_facecolor('black')
        fig.set_size_inches(w=dim_size[dim_remaining[i][0]]*2, h=dim_size[dim_remaining[i][1]]*2, forward=False)
        ax = plt.gca()
        ax.set_facecolor('black')
        shift_zero = -(off_origin[dim_remaining[i][1]] - dim_size[dim_remaining[i][1]] //2)
        y_pos = dim_size[dim_remaining[i][1]] - (xy_filt[:, 1] + 2 * shift_zero) # Invert y-axis for plotting
        plt.scatter(xy_filt[:, 0], y_pos, s=size, c=color, marker='o', alpha=alpha)

        plt.xlim(off_origin[dim_remaining[i][0]] - dim_size[dim_remaining[i][0]] / 2, 
                 off_origin[dim_remaining[i][0]] + dim_size[dim_remaining[i][0]] / 2)
        plt.ylim(off_origin[dim_remaining[i][1]] - dim_size[dim_remaining[i][1]] / 2,
                 off_origin[dim_remaining[i][1]] + dim_size[dim_remaining[i][1]] / 2)
        plt.axis('off')
        plt.tight_layout()
        suffix = f"slice_pos{center_loc}_{dim_names[i]}"
        plt.savefig(f"{save_path}{suffix}.png", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI,facecolor=fig.get_facecolor())
        plt.close()

def save_error_maps(save_path, vol, gt_vol):
    """Save absolute-error heatmaps for the three mid-slices.

    Args:
        save_path: Base path prefix for PNG exports.
        vol: Predicted volume array/tensor.
        gt_vol: Ground-truth volume array/tensor.
    """
    vol = np.squeeze(ensure_numpy(vol)).astype(np.float32)
    gt_vol = np.squeeze(ensure_numpy(gt_vol)).astype(np.float32)

    vol = np.clip(vol, 0.0, 1.0)
    gt_vol = np.clip(gt_vol, 0.0, 1.0)

    center_loc = np.array(vol.shape) // 2
    mid_slices_vol = [
        vol[center_loc[0], :, :],
        vol[:, center_loc[1], :],
        vol[:, :, center_loc[2]],
    ]
    mid_slices_gt = [
        gt_vol[center_loc[0], :, :],
        gt_vol[:, center_loc[1], :],
        gt_vol[:, :, center_loc[2]],
    ]

    dim_names = ["YZ", "XZ", "XY"]
    for i in range(3):
        slice_gt = np.repeat(mid_slices_gt[i][:,:,None], 3, axis=-1)
        slice_vol = np.repeat(mid_slices_vol[i][:,:,None], 3, axis=-1)

        suffix = f"slice_id{center_loc[i]}_{dim_names[i]}"
        error_map = np.abs(slice_gt - slice_vol)
        error_map = (error_map * 255.0).astype(np.uint8)
        skimage.io.imsave(f"{save_path}{suffix}.png", error_map, check_contrast=False)
