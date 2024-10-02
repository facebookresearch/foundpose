#!/usr/bin/env python3

from itertools import product
from typing import Any, Dict, Tuple, NamedTuple, Mapping, Sequence, Optional, List

import torch

import numpy as np

from utils.misc import (
    array_to_tensor,
    arrays_to_tensors,
)
from bop_toolkit_lib import inout
from utils.structs import AlignedBox2f, PinholePlaneCameraModel

from utils import json_util, logging, structs, misc

logger: logging.Logger = logging.get_logger()

class DatasetOpts(NamedTuple):
    """Options that can be specified via the command line.

    crop_image_center: Whether to crop the center of each image.
    center_crop_size: Size of the region to crop from the center of each image.
    add_depth: Whether to include depth images.
    num_workers: The number of workers.
    """

    crop_image_center: bool = True# False
    center_crop_size: Tuple[int, int] = (630, 476)  # For DINOv2 with 14x14 patches.
    add_depth: bool = True
    convert_rgb_to_mono: bool = False


def prepare_sample(
    sample_info: Mapping[str, Any],
    split_props: Mapping[str, Any],
    chunk_cameras: Mapping[int, Mapping[int, PinholePlaneCameraModel]],
    chunk_gts: Mapping[int, Mapping[int, Sequence[structs.ObjectAnnotation]]],
    chunk_gts_info: Mapping[int, Mapping[int, Sequence[Mapping[str, Any]]]],
    # opts: DatasetOpts,
) -> Dict[str, Any]:
    """Produces a packed FrameSequence with a single image and its annotations.

    Args:
        sample_info: A dictionary with IDs of the dataset chunk ("chunk_id") and
            the image ("im_id").
        split_props: Properties of the dataset split.
        chunk_cameras: A dictionary mapping chunk/image IDs to camera properties.
        chunk_gts: A dictionary mapping chunk/image IDs to GT annotations.
        chunk_gts_info: A dictionary mapping chunk/image IDs to meta information
            about GT annotations.
        opts: Options of this script.
    Returns:
        A packed FrameSequence represented as a dictionary that can be msgpacked.
    """

    opts = DatasetOpts()

    # IDs of the dataset chunk and the image.
    chunk_id, im_id = sample_info["scene_id"], sample_info["im_id"]

    # Camera parameters.
    camera = chunk_cameras[chunk_id][im_id]

    # Cropping of the image center.
    center_crop_box = None
    if opts.crop_image_center:
        (
            crop_model,
            center_crop_box,
        ) = construct_center_crop_camera(
            camera_model_orig=camera,
            center_crop_size=opts.center_crop_size,
        )
        camera = crop_model

    # Load the image.
    if "gray" in split_props["im_modalities"]:
        # ITODD includes grayscale instead of RGB images.
        image_path = split_props["gray_tpath"].format(scene_id=chunk_id, im_id=im_id)
    else:
        image_path = split_props["rgb_tpath"].format(scene_id=chunk_id, im_id=im_id)

    image = inout.load_im(image_path)
    if image.ndim == 3 and opts.convert_rgb_to_mono:
        image = np.expand_dims(misc.rgb_to_mono_image(image), -1)
    elif image.ndim == 2:
        image = np.expand_dims(image, -1)

    # Load the depth image.
    depth_image = None
    if opts.add_depth:
        depth_path = split_props["depth_tpath"].format(scene_id=chunk_id, im_id=im_id)
        depth_image = np.array(inout.load_depth(depth_path), dtype=np.float32)

    # Simple cropping of the image center.
    if opts.crop_image_center and center_crop_box is not None:
        image = misc.crop_image(image, center_crop_box)
        if depth_image is not None:
            depth_image = misc.crop_image(depth_image, center_crop_box)

    # Object annotations.
    objects_anno = None
    if len(chunk_gts) and len(chunk_gts_info) and len(chunk_gts[chunk_id][im_id]):
        objects_anno = []
        for gt_id, gt in enumerate(chunk_gts[chunk_id][im_id]):
            gt_info = chunk_gts_info[chunk_id][im_id][gt_id]

            # Load and encode the object mask.
            mask_modal_path = split_props["mask_visib_tpath"].format(
                scene_id=chunk_id, im_id=im_id, gt_id=gt_id
            )
            mask_modal = inout.load_im(mask_modal_path) / 255.0

            # 2D (amodal) bounding box of the object.
            box = gt_info["bbox_obj"]
            box_amodal = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

            # Visibility of the object.
            visibility = gt_info["visib_fract"]

            # 6D object pose.
            pose_m2w = None
            if gt.pose is not None:
                pose_m2c = gt.pose
                trans_c2w = camera.T_world_from_eye
                trans_m2w = np.matmul(trans_c2w, misc.get_rigid_matrix(pose_m2c))
                pose_m2w = structs.ObjectPose(R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:])

            # Update annotations in case of cropping.
            if opts.crop_image_center and center_crop_box is not None:
                mask_modal_orig = np.array(mask_modal)
                mask_modal = misc.crop_image(mask_modal, center_crop_box)

                box_amodal[0] -= center_crop_box.left
                box_amodal[1] -= center_crop_box.top
                box_amodal[2] -= center_crop_box.left
                box_amodal[3] -= center_crop_box.top

                visibility *= np.sum(mask_modal) / np.sum(mask_modal_orig)

            objects_anno.append(
                structs.ObjectAnnotation(
                    dataset=split_props["name"],
                    lid=gt.lid,
                    pose=pose_m2w,
                    boxes_amodal=np.array(box_amodal),
                    masks_modal=np.array(mask_modal, dtype=np.uint8),
                    visibilities=np.asarray(visibility),
                )
            )

    # Convert the frame sequence to a dictionary with encoded images.
    return structs.SceneAnnotation(
            image=np.array(image),
            depth_image=np.array(depth_image),
            camera=camera,
            objects_anno=objects_anno,
        )


def parse_chunk_cameras(
    cameras_json: Any,
    im_size: Optional[Tuple[int, int]] = None,
) -> Dict[int, PinholePlaneCameraModel]:
    """Parses per-image camera parameters of a dataset chunk from JSON format.

    Args:
        cameras_json: Cameras in JSON format.
        im_size: The image size (needs to be either provided via this argument
            or must be present in the JSON file).
    Returns:
        A dictionary mapping the image ID to camera parameters.
        For each image, there is a dictionary with the following items:
    """

    cameras = {}
    for im_id, camera_json in cameras_json.items():

        width = None
        height = None
        fx = None
        fy = None
        cx = None
        cy = None
        depth_scale = None

        # World to camera transformation.
        extrinsics_w2c = np.eye(4, dtype=np.float32)

        if im_size is not None:
            width = im_size[0]
            height = im_size[1]

        for k, v in camera_json.items():
            if k == "im_size":
                width = int(v[0])
                height = int(v[1])
            elif k == "cam_K":
                K = np.array(v, np.float32).reshape((3, 3))
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            elif k == "cam_R_w2c":
                extrinsics_w2c[:3, :3] = np.array(v, np.float32).reshape((3, 3))
            elif k == "cam_t_w2c":
                extrinsics_w2c[:3, 3:] = np.array(v, np.float32).reshape((3, 1))
            elif k == "depth_scale":
                depth_scale = float(v)

        # Camera to world transformation.
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)

        camera_model = PinholePlaneCameraModel(
            width=width,
            height=height,
            f=(fx, fy),
            c=(cx, cy),
            T_world_from_eye=extrinsics_c2w,
        )
        cameras[im_id] = camera_model

    return cameras

def load_chunk_cameras(
    path: str, im_size: Optional[Tuple[int, int]] = None
) -> Dict[int, PinholePlaneCameraModel]:
    """Loads per-image camera parameters of a dataset chunk from a JSON file.

    Args:
        path: The path to the input JSON file.
        im_size: The image size (needs to be either provided via this argument
            or must be present in the JSON file).
    Returns:
        A dictionary mapping the image ID to camera parameters.
        For each image, there is a dictionary with the following items:
    """

    cameras_json = json_util.load_json(path, keys_to_int=True)

    return parse_chunk_cameras(cameras_json, im_size)

def parse_chunk_gts(
    gts_json: Any,
    dataset: str,
) -> Dict[int, List[structs.ObjectAnnotation]]:
    """Parses GT annotations of a dataset chunk from JSON format.

    Args:
        gts_json: Chunk GTs in JSON format.
        dataset: A dataset which the JSON file belongs to.
    Returns:
        A dictionary with the loaded GT annotations.
    """

    gts = {}
    for im_id, im_gts_json in gts_json.items():
        gts[im_id] = []
        for gt_raw in im_gts_json:
            dataset_curr, lid, R, t = None, None, None, None
            for k, v in gt_raw.items():
                if k == "dataset":
                    dataset_curr = str(v)
                if k == "obj_id":
                    lid = int(v)
                elif k == "cam_R_m2c":
                    R = np.array(v, np.float32).reshape((3, 3))
                elif k == "cam_t_m2c":
                    t = np.array(v, np.float32).reshape((3, 1))
            if dataset_curr is None:
                dataset_curr = dataset
            if lid is None:
                raise ValueError("Local ID must be specified.")
            gts[im_id].append(
                structs.ObjectAnnotation(
                    dataset=dataset_curr, lid=lid, pose=structs.ObjectPose(R=R, t=t)
                )
            )
    return gts

def load_chunk_gts(
    path: str, dataset: str
) -> Dict[int, List[structs.ObjectAnnotation]]:
    """Loads GT annotations of a dataset chunk from a JSON file.

    Args:
        path: The path to the input JSON file.
        dataset: A dataset which the JSON file belongs to.
    Returns:
        A dictionary with the loaded GT annotations.
    """

    gts_json = json_util.load_json(path, keys_to_int=True)

    return parse_chunk_gts(gts_json, dataset)


def construct_center_crop_camera(
    camera_model_orig: PinholePlaneCameraModel,
    center_crop_size: Tuple[int, int],
) -> Tuple[PinholePlaneCameraModel, AlignedBox2f]:
    """Constructs a virtual camera focused on the viewport center.

    Args:
        camera_model_orig: Original camera model.
        center_crop_size: Size of the region to crop from the center of the
            camera viewport.
    Returns:
        camera_model: A virtual camera whose viewport has the specified size and
        is centered on the original camera viewport. The camera distortion is the
        same as for the original camera.
        center_crop_box: A box corresponding to the new camera viewport.
    """

    if (
        center_crop_size[0] > camera_model_orig.width
        or center_crop_size[1] > camera_model_orig.height
    ):
        raise ValueError(
            "The center crop cannot be larger than the original camera viewport."
        )

    camera_model = camera_model_orig.copy()

    camera_model.width = center_crop_size[0]
    camera_model.height = center_crop_size[1]

    left = int(0.5 * (camera_model_orig.width - camera_model.width))
    top = int(0.5 * (camera_model_orig.height - camera_model.height))
    camera_model.c = (camera_model_orig.c[0] - left, camera_model_orig.c[1] - top)

    center_crop_box = AlignedBox2f(
        left, top, left + center_crop_size[0], top + center_crop_size[1]
    )

    return camera_model, center_crop_box
