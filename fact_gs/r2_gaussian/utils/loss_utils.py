#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

from fused_3d_tv import tv3d_loss


def tv_3d_loss(vol, reduction_value=1):
    # Replaced manual TV loss calculation with fused_tv3d
    tv = tv3d_loss(vol.clone().unsqueeze(0).unsqueeze(0))

    return tv/reduction_value


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


