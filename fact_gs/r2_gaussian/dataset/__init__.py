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
import sys
import random
import numpy as np
import os.path as osp
import torch
import tifffile

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel
from fact_gs.r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from fact_gs.r2_gaussian.utils.camera_utils import cameraList_from_camInfos
from fact_gs.r2_gaussian.utils.general_utils import t2a


class SceneRecon:
    gaussians: GaussianModel

    def __init__(
        self,
        model_args,
        shuffle=True,
    ):
        self.model_path = model_args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        if osp.exists(osp.join(model_args.data_source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                model_args.data_source_path,
                model_args.eval,
            )
        elif model_args.data_source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                model_args.data_source_path,
                model_args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {model_args.data_source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, model_args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, model_args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, step, queryfunc, vol_format="npy"):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/step_{}".format(step)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            if vol_format == "npy":
                np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                    t2a(vol_pred),
                )
            elif vol_format == "tiff":
                tifffile.imwrite(osp.join(point_cloud_path, "vol_pred.tiff"), t2a(vol_pred))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras


class SceneVol:
    gaussians: GaussianModel

    def __init__(
        self,
        model_args,
        shuffle=True,
        file_name="vol_prior",
    ):
        self.model_path = model_args.model_path
        self.file_name = file_name
        
        # Read scene info
        if osp.exists(osp.join(model_args.data_source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["BlenderVol"](
                model_args.data_source_path,
                file_name=self.file_name,
            )
        elif model_args.data_source_path.split(".")[-1] in ["pickle", "pkl"]:
            raise NotImplementedError("NAF format not implemented for volume compression.")
        else:
            assert False, f"Could not recognize scene type: {model_args.data_source_path}."


        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, step, queryfunc, vol_format="npy"):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/step_{}".format(step)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            if vol_format == "npy":
                np.save(
                    osp.join(point_cloud_path, "vol_pred.npy"),
                    t2a(vol_pred),
                )
            elif vol_format == "tiff":
                tifffile.imwrite(osp.join(point_cloud_path, "vol_pred.tiff"), t2a(vol_pred))
