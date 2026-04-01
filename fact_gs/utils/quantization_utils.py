import torch


def ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    """Quantize a tensor with a straight-through estimator.

    Args:
        x: Tensor to be quantized.
        num_bits: Bit depth of the quantizer.

    Returns:
        Tensor with quantized values but preserved gradients.
    """
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = x.min().item(), x.max().item()
    scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
    # Quantize in forward pass (non-differentiable)
    q_x = torch.round((x - min_val) / scale).clamp(qmin, qmax)
    dq_x = q_x * scale + min_val
    # Restore gradients in backward pass
    dq_x = x + (dq_x - x).detach()
    return dq_x

def quantize_gaussians(gaussians, optim_args):
    """Quantize Gaussian parameters in-place according to optimizer settings.

    Args:
        gaussians: Gaussian model whose tensors will be quantized.
        optim_args: Optimizer config node that stores per-field bit widths.
    """
    with torch.no_grad():
        gaussians._xyz.copy_(ste_quantize(gaussians._xyz, optim_args.pos_bits))
        gaussians._scaling.copy_(ste_quantize(gaussians._scaling, optim_args.scale_bits))
        gaussians._rotation.copy_(ste_quantize(gaussians._rotation, optim_args.rot_bits))
        gaussians._density.copy_(ste_quantize(gaussians._density, optim_args.feat_bits))


def report_model_size(gaussians, optim_args):
    """Print the approximate model memory footprint under current precision.

    Args:
        gaussians: Gaussian model exposing ``get_xyz`` and ``_density`` tensors.
        optim_args: Optimizer config determining whether quantization is active.
    """
    if optim_args.quantize:
        pos_bytes = optim_args.pos_bits / 8
        scale_bytes = optim_args.scale_bits / 8
        rot_bytes = optim_args.rot_bits / 8
        feat_bytes = optim_args.feat_bits / 8
    else:
        pos_bytes = 4  # float32
        scale_bytes = 4
        rot_bytes = 4
        feat_bytes = 4
    total_bytes_per_gaussian = 3 * pos_bytes + 3 * scale_bytes + 4 * rot_bytes + gaussians._density.shape[1] * feat_bytes
    total_size_mb = (total_bytes_per_gaussian * gaussians.get_xyz.shape[0]) / (1024 * 1024)
    print(f"Model size: {total_size_mb:.2f} MB with {gaussians.get_xyz.shape[0]} Gaussians.")
