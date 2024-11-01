#!/usr/bin/env python3

"""Miscellaneous functions."""

import dataclasses
from dataclasses import asdict
import math
import time
import cv2
import uuid

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping
from collections import namedtuple

import numpy as np
from PIL import Image
import torch

from utils import geometry, logging

import utils.structs as structs
from utils.structs import AlignedBox2f, CameraModel, PinholePlaneCameraModel

from utils.geometry import transform_3d_points_numpy, gen_look_at_matrix

logger: logging.Logger = logging.get_logger()


class Timer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_time = None

    def start(self):
        if self.enabled:
            self.start_time = time.time()

    def elapsed(self, msg="Elapsed") -> Optional[float]:
        if self.enabled:
            elapsed = time.time() - self.start_time
            logger.info(f"{msg}: {elapsed:.5f}s")
            return elapsed
        else:
            return None

def fibonacci_sampling(
    n_pts: int, radius: float = 1.0
) -> List[Tuple[float, float, float]]:
    """Fibonacci-based sampling of points on a sphere.

    Samples an odd number of almost equidistant 3D points from the Fibonacci
    lattice on a unit sphere.

    Ref:
    [1] https://arxiv.org/pdf/0912.4540.pdf
    [2] http://stackoverflow.com/questions/34302938/map-point-to-closest-point-on-fibonacci-lattice
    [3] http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    [4] https://www.openprocessing.org/sketch/41142

    Args:
        n_pts: Number of 3D points to sample (an odd number).
        radius: Radius of the sphere.
    Returns:
        List of 3D points on the sphere surface.
    """

    # Needs to be an odd number [1].
    assert n_pts % 2 == 1

    n_pts_half = int(n_pts / 2)

    phi = (math.sqrt(5.0) + 1.0) / 2.0  # Golden ratio.
    phi_inv = phi - 1.0
    ga = 2.0 * math.pi * phi_inv  # Complement to the golden angle.

    pts = []
    for i in range(-n_pts_half, n_pts_half + 1):
        lat = math.asin((2 * i) / float(2 * n_pts_half + 1))
        lon = (ga * i) % (2 * math.pi)

        # Convert the latitude and longitude angles to 3D coordinates.
        # Latitude (elevation) represents the rotation angle around the X axis.
        # Longitude (azimuth) represents the rotation angle around the Z axis.
        s = math.cos(lat) * radius
        x, y, z = math.cos(lon) * s, math.sin(lon) * s, math.tan(lat) * s
        pts.append([x, y, z])

    return pts

def sample_views(
    min_n_views: int,
    radius: float = 1.0,
    azimuth_range: Tuple[float, float] = (0, 2 * math.pi),
    elev_range: Tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi),
    mode: str = "fibonacci",
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Viewpoint sampling from a view sphere.

    Args:
        min_n_views: The min. number of points to sample on the whole sphere.
        radius: Radius of the sphere.
        azimuth_range: Azimuth range from which the viewpoints are sampled.
        elev_range: Elevation range from which the viewpoints are sampled.
        mode: Type of sampling (options: "fibonacci").
    Returns:
        List of views, each represented by a 3x3 ndarray with a rotation
        matrix and a 3x1 ndarray with a translation vector.
    """

    # Get points on a sphere.
    if mode == "fibonacci":
        n_views = min_n_views
        if n_views % 2 != 1:
            n_views += 1

        pts = fibonacci_sampling(n_views, radius=radius)
        pts_level = [0 for _ in range(len(pts))]
    else:
        raise ValueError("Unknown view sampling mode.")

    views = []
    for pt in pts:
        # Azimuth from (0, 2 * pi).
        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        # Elevation from (-0.5 * pi, 0.5 * pi).
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        if not (
            azimuth_range[0] <= azimuth <= azimuth_range[1]
            and elev_range[0] <= elev <= elev_range[1]
        ):
            continue

        # Rotation matrix.
        # Adopted from gluLookAt function (uses OpenGL coordinate system):
        # [1] http://stackoverflow.com/questions/5717654/glulookat-explanation
        # [2] https://www.opengl.org/wiki/GluLookAt_code
        f = -np.array(pt)  # Forward direction.
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0])  # Up direction.
        s = np.cross(f, u)  # Side direction.
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis.
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)  # Recompute up.
        R = np.array([[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]])

        # Convert from OpenGL to OpenCV coordinate system.
        R_yz_flip = geometry.rotation_matrix_numpy(math.pi, np.array([1, 0, 0]))[
            :3, :3
        ]
        R = R_yz_flip.dot(R)

        # Translation vector.
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({"R": R, "t": t})

    return views, pts_level


def calc_crop_box(
    box: AlignedBox2f,
    box_scaling_factor: float = 1.0,
    make_square: bool = False,
) -> AlignedBox2f:
    """Adjusts a bounding box to the specified aspect and scale.

    Args:
        box: Bounding box.
        box_aspect: The aspect ratio of the target box.
        box_scaling_factor: The scaling factor to apply to the box.
    Returns:
        Adjusted box.
    """

    # Potentially inflate the box and adjust aspect ratio.
    crop_box_width = box.width * box_scaling_factor
    crop_box_height = box.height * box_scaling_factor

    # Optionally make the box square.
    if make_square:
        crop_box_side = max(crop_box_width, crop_box_height)
        crop_box_width = crop_box_side
        crop_box_height = crop_box_side

    # Calculate padding.
    x_pad = 0.5 * (crop_box_width - box.width)
    y_pad = 0.5 * (crop_box_height - box.height)

    return AlignedBox2f(
        left=box.left - x_pad,
        top=box.top - y_pad,
        right=box.right + x_pad,
        bottom=box.bottom + y_pad,
    )


def construct_crop_camera(
    box: AlignedBox2f,
    camera_model_c2w: CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float,
) -> CameraModel:
    """Constructs a virtual pinhole camera from the specified 2D bounding box.

    Args:
        camera_model_c2w: Original camera model with extrinsics set to the
            camera->world transformation.

        viewport_crop_size: Viewport size of the new camera.
        viewport_scaling_factor: Requested scaling of the viewport.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center
        of the specified 2D bounding box and whose focal length is set such as
        the sphere representing the bounding box (+ requested padding) is visible
        in the camera viewport.
    """

    # Get centroid and radius of the reference sphere (the virtual camera will
    # be constructed such as the projection of the sphere fits the viewport.
    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    box_corners_in_c = np.array(
        [
            [box.left - cx, box.top - cy, f],
            [box.right - cx, box.top - cy, f],
            [box.left - cx, box.bottom - cy, f],
            [box.right - cx, box.bottom - cy, f],
        ]
    )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[:3, 0]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    # Transformations from world to the original and virtual cameras.
    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = gen_look_at_matrix(trans_w2c, centroid_in_w)

    # Transform the centroid from world to the virtual camera.
    centroid_in_vc = transform_3d_points_numpy(
        trans_w2vc, np.expand_dims(centroid_in_w, axis=0)
    ).squeeze()

    # Project the sphere radius to the image plane of the virtual camera and
    # enlarge it by the specified padding. This defines the 2D extent that
    # should be visible in the virtual camera.
    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = fx_fy_orig * cx_cy / extent_2d

    # Parameters of the virtual camera.
    return PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )

def calc_2d_box(
    xs: torch.Tensor,
    ys: torch.Tensor,
    im_size: Optional[torch.Tensor] = None,
    clip: bool = False,
) -> torch.Tensor:
    """Calculates the 2D bounding box of a set of 2D points.

    Args:
        xs: A 1D tensor with x-coordinates of 2D points.
        ys: A 1D tensor with y-coordinates of 2D points.
        im_size: The image size (width, height), used for optional clipping.
        clip: Whether to clip the bounding box (default == False).
    Returns:
        The 2D bounding box (x1, y1, x2, y2), where (x1, y1) and (x2, y2) is the
        minimum and the maximum corner respectively.
    """
    if len(xs) == 0 or len(ys) == 0:
        return torch.Tensor([0.0, 0.0, 0.0, 0.0])

    box_min = torch.as_tensor([xs.min(), ys.min()])
    box_max = torch.as_tensor([xs.max(), ys.max()])
    if clip:
        if im_size is None:
            raise ValueError("Image size needs to be provided for clipping.")
        box_min = clip_2d_point(box_min, im_size)
        box_max = clip_2d_point(box_max, im_size)
    return torch.hstack([box_min, box_max])


def get_rigid_matrix(trans: structs.RigidTransform) -> np.ndarray:
    """Creates a 4x4 transformation matrix from a 3x3 rotation and 3x1 translation.

    Args:
        trans: A rigid transformation defined by a 3x3 rotation matrix and
            a 3x1 translation vector.
    Returns:
        A 4x4 rigid transformation matrix.
    """

    matrix = np.eye(4)
    matrix[:3, :3] = trans.R
    matrix[:3, 3:] = trans.t
    return matrix


def get_intrinsic_matrix(cam: CameraModel) -> np.ndarray:
    """Returns a 3x3 intrinsic matrix of the given camera.

    Args:
        cam: The input camera model.
    Returns:
        A 3x3 intrinsic matrix K.
    """

    return np.array(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ]
    )

def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:
    """Resizes an image.

    Args:
      image: An input image.
      size: The size of the output image (width, height).
      interpolation: An interpolation method (a suitable one is picked if undefined).
    Returns:
      The resized image.
    """

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
        )
    return cv2.resize(image, size, interpolation=interpolation)


def map_fields(func, obj, only_type=object):
    """
    map 'func' recursively over nested collection types.

    >>> map_fields(lambda x: x * 2,
    ...            {'a': 1, 'b': {'x': 2, 'y': 3}})
    {'a': 2, 'b': {'x': 4, 'y': 6}}

    E.g. to detach all tensors in a network output frame:

        frame = map_fields(torch.detach, frame, torch.Tensor)

    The optional 'only_type' parameter only calls `func` for values where
    isinstance(value, only_type) returns True. Other values are returned
    as-is.
    """
    if is_dictlike(obj):
        ty = type(obj)
        if isinstance(obj, Mapping):
            return ty((k, map_fields(func, v, only_type)) for (k, v) in obj.items())
        else:
            # NamedTuple or dataclass
            return ty(
                **{k: map_fields(func, v, only_type) for (k, v) in asdict(obj).items()}
            )
    elif isinstance(obj, tuple):
        return tuple(map_fields(func, v, only_type) for v in obj)
    elif isinstance(obj, list):
        return [map_fields(func, v, only_type) for v in obj]
    elif isinstance(obj, only_type):
        return func(obj)
    else:
        return obj

def is_dictlike(obj: Any) -> bool:
    """
    Returns true if the object is a dataclass, NamedTuple, or Mapping.
    """
    return (
        dataclasses.is_dataclass(obj)
        or hasattr(obj, "_asdict")
        or isinstance(obj, Mapping)
    )

def chw_to_hwc(data: np.ndarray) -> np.ndarray:
    """Converts a Numpy array from CHW to HWC (C = channels, H = height, W = width).

    Args:
        data: A Numpy array width dimensions in the CHW order.
    Returns:
        A Numpy array width dimensions in the HWC order.
    """

    return np.transpose(data, (1, 2, 0))

def slugify(string: str) -> str:
    """Slugify a string (typically a path) such as it can be used as a filename.

    Args:
        string: A string to slugify.
    Returns:
        A slugified string.
    """
    return string.strip("/").replace("/", "-").replace(" ", "-").replace(".", "-")

def crop_image(image: np.ndarray, crop_box: AlignedBox2f) -> np.ndarray:
    """Crops an image.

    Args:
        image: The input HWC image.
        crop_box: The bounding box for cropping given by (x1, y1, x2, y2).
    Returns:
        Cropped image.
    """

    return image[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]


def ensure_three_channels(im: np.ndarray) -> np.ndarray:
    """Ensures that the image has 3 channels.

    Args:
        im: The input image.
    Returns:
        An image with 3 channels (single-channel images are duplicated).
    """

    if im.ndim == 3:
        return im
    elif im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        return np.dstack([im, im, im])
    else:
        raise ValueError("Unknown image format.")

def warp_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
    factor_to_downsample: int = 1,
) -> np.ndarray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    factor_to_downsample :
        If this value is greater than 1, it will downsample the input image prior to warping.
        This improves downsampling performance, in an attempt to replicate
        area interpolation for crop+undistortion warps.
    """

    if factor_to_downsample > 1:
        src_image = cv2.resize(
            src_image,
            (
                int(src_image.shape[1] / factor_to_downsample),
                int(src_image.shape[0] / factor_to_downsample),
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Rescale source camera
        src_camera = adjust_camera_model(src_camera, factor_to_downsample)

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def warp_depth_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:

    # Copy the source depth image.
    depth_image = np.array(src_depth_image)

    # If the camera extrinsics changed, update the depth values.
    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):

        # Image coordinates with valid depth values.
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        # Transform the source depth image to a point cloud.
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)

        # Transform the point cloud from the source to the target camera.
        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    # Warp the depth image to the target camera.
    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )


def array_to_tensor(
    array: np.ndarray, make_array_writeable: bool = True
) -> torch.Tensor:
    """Converts a Numpy array into a tensor.

    Args:
        array: A Numpy array.
        make_array_writeable: Whether to force the array to be writable.
    Returns:
        A tensor.
    """

    # If the array is not writable, make it writable or copy the array.
    # Otherwise, torch.from_numpy() would yield a warning that tensors do not
    # support the writing lock and one could modify the underlying data via them.
    if not array.flags.writeable:
        if make_array_writeable and array.flags.owndata:
            array.setflags(write=True)
        else:
            array = np.array(array)
    return torch.from_numpy(array)

def arrays_to_tensors(data: Any) -> Any:
    """Recursively converts Numpy arrays into tensors.

    Args:
        data: A possibly nested structure with Numpy arrays.
    Returns:
        The same structure but with Numpy arrays converted to tensors.
    """

    return map_fields(lambda x: array_to_tensor(x), data, only_type=np.ndarray)


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor into a Numpy array.

    Args:
        tensor: A tensor (may be in the GPU memory).
    Returns:
        A Numpy array.
    """

    return tensor.detach().cpu().numpy()


def tensors_to_arrays(data: Any) -> Any:
    """Recursively converts tensors into Numpy arrays.

    Args:
        data: A possibly nested structure with tensors.
    Returns:
        The same structure but with tensors converted to Numpy arrays.
    """

    return map_fields(
        lambda x: tensor_to_array(x), data, only_type=torch.Tensor)


