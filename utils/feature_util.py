#!/usr/bin/env python3

from typing import List, Tuple

import kornia
import torch
import torch.nn.functional as F

from utils import logging, misc, geometry
from utils.structs import PinholePlaneCameraModel

from utils import dinov2_utils

logger: logging.Logger = logging.get_logger()


def make_feature_extractor(model_name: str) -> torch.nn.Module:

    if model_name.startswith("dinov2_"):
        return dinov2_utils.DinoFeatureExtractor(model_name=model_name)
    else:
        raise NotImplementedError(model_name)

def generate_grid_points(
    grid_size: Tuple[int, int],
    cell_size: float = 1.0,
) -> torch.Tensor:
    """Generates 2D coordinates at the centers of cells of a regular grid.
    Args:
        grid_size: Size of the grid expressed as (grid_width, grid_height).
        cell_size: Size of a cell expressed as a single number as cell are square.
    Returns:
        Generated 2D coordinates.
    """

    # Width and height of the grid expressed in the number of cells.
    grid_cols = int(grid_size[0] / cell_size)
    grid_rows = int(grid_size[1] / cell_size)

    # Generate 2D coordinates at the centers of the grid cells.
    cell_half_size = cell_size / 2.0
    x = torch.linspace(
        cell_half_size, grid_size[0] - cell_half_size, grid_cols, dtype=torch.float
    )
    y = torch.linspace(
        cell_half_size, grid_size[1] - cell_half_size, grid_rows, dtype=torch.float
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    # 2D coordinates of shape (num_points, 2).
    return torch.vstack((grid_x.flatten(), grid_y.flatten())).T


def filter_points_by_box(
    points: torch.Tensor, box: Tuple[float, float, float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keeps only points inside the specified 2D bounding box.

    Args:
        points: 2D coordinates of shape (num_points, 2).
        box: A bounding box expressed as (x1, y1, x2, y2).
    Returns:
        Filtered points and the mask.
    """

    x1, y1, x2, y2 = box
    valid_mask = torch.logical_and(
        torch.logical_and(points[:, 0] > x1, points[:, 0] < x2),
        torch.logical_and(points[:, 1] > y1, points[:, 1] < y2),
    )
    return points[valid_mask], valid_mask


def filter_points_by_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keeps only points inside the specified mask.

    Args:
        points: 2D coordinates of shape (num_points, 2).
        mask: A binary mask.
    Returns:
        Filtered points.
    """

    # Convert the integers so we can use the coordinates for indexing.
    # Add 0.5 to convert to image coordinates before masking.
    points_int = (points + 0.5).int()

    # Keep only points inside the canvas.
    points_int, valid_mask = filter_points_by_box(
        points_int, (0, 0, mask.shape[1], mask.shape[0])
    )

    # Keep only points inside the mask.
    points = points[valid_mask][mask[points_int[:, 1], points_int[:, 0]].bool()]

    return points


def sample_feature_map_at_points(
    feature_map_chw: torch.Tensor, points: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    """Samples a feature map at the specified 2D coordinates.

    Args:
        feature_map_chw: A tensor of shape (C, H, W).
        points: A tensor of shape (N, 2) where N is the number of points.
        image_size: Size of the input image expressed as (image_width, image_height).
            2D coordinates of the points are expressed in the image coordinates.
    Returns:
        A tensor of shape (num_points, feature_dim) containing the sampled
        features at given 2D coordinates.
    """

    # Normalize the 2D coordinates to [-1, 1].
    uv = torch.div(2.0, torch.as_tensor(image_size)).to(points.device) * points - 1.0

    # Convert the 2D coordinates to shape (1, N, 1, 2).
    query_coords = uv.unsqueeze(0).unsqueeze(2)

    # Feature vectors of shape [1, C, N, 1].
    features = torch.nn.functional.grid_sample(
        feature_map_chw.unsqueeze(0),
        query_coords,
        align_corners=False,
    )

    # Reshape the feature vectors to (N, C).
    features = features[0, :, :, 0].permute(1, 0)

    return features


def lift_2d_points_to_3d(
    points: torch.Tensor,
    depth_image: torch.Tensor,
    camera_model: PinholePlaneCameraModel,
) -> torch.Tensor:
    device = points.device

    # The considered focal length is the average of fx and fy.
    focal = 0.5 * (camera_model.f[0] + camera_model.f[1])

    # 3D points in the camera space.
    points_3d_in_cam = torch.hstack(
        [
            points - torch.as_tensor(camera_model.c).to(torch.float32).to(device),
            focal * torch.ones(points.shape[0], 1).to(torch.float32).to(device),
        ]
    )
    depths = depth_image[
        torch.floor(points[:, 1]).to(torch.int32),
        torch.floor(points[:, 0]).to(torch.int32),
    ].reshape(-1, 1)
    points_3d_in_cam *= depths / points_3d_in_cam[:, 2].reshape(-1, 1)

    return points_3d_in_cam


def get_visual_features_registered_in_3d(
    image_chw: torch.Tensor,
    depth_image_hw: torch.Tensor,
    object_mask: torch.Tensor,
    camera: PinholePlaneCameraModel,
    T_model_from_camera: torch.Tensor,
    extractor: torch.nn.Module,
    grid_cell_size: float,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = image_chw.device

    timer = misc.Timer(enabled=debug)
    timer.start()

    # Generate grid points at which to sample feature vectors.
    grid_points = generate_grid_points(
        grid_size=(image_chw.shape[2], image_chw.shape[1]),
        cell_size=grid_cell_size,
    ).to(device)

    # Erode the mask a bit to ignore pixels at the contour where
    # depth values tend to be noisy.
    kernel = torch.ones(5, 5).to(device)
    object_mask_eroded = (
        kornia.morphology.erosion(
            object_mask.reshape(1, 1, *object_mask.shape).to(torch.float32), kernel
        )
        .squeeze([0, 1])
        .to(object_mask.dtype)
    )

    # Keep only grid points inside the object mask.
    query_points = filter_points_by_mask(grid_points, object_mask_eroded)

    # Get 3D coordinates corresponding to the query points.
    vertices_in_cam = lift_2d_points_to_3d(
        points=query_points,
        depth_image=depth_image_hw,
        camera_model=camera,
    )
    # Transform vertices to the model space.
    vertices_in_model = geometry.transform_3d_points_torch(
        T_model_from_camera.to(device), vertices_in_cam.to(device)
    )
    vertex_ids = torch.arange(vertices_in_model.shape[0], dtype=torch.int32)

    timer.elapsed("Time for preparation")

    # Extract feature vectors.
    timer.start()
    image_bchw = image_chw.unsqueeze(0)

    timer.start()

    # Extract feature map at the current image scale.
    extractor_output = extractor(image_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]
    feature_map_chw = feature_map_chw.to(device)

    timer.elapsed(f"Time for feature extraction")
    timer.start()

    # Extract feature vectors at query points.
    feat_vectors = sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        image_size=(image_chw.shape[-1], image_chw.shape[-2]),
    ).detach()

    timer.elapsed(f"Time for feature sampling.")

    return (
        feat_vectors,
        vertex_ids,
        vertices_in_model,
    )
