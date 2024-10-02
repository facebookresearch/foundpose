#!/usr/bin/env python3

import math
import numpy as np
import torch
from typing import Tuple, TypeVar
import scipy
from utils import geometry

from scipy.spatial.transform import Rotation

AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")

def transform_3d_points_numpy(trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform 3D points. Compute trans * points

    Args:
        points: 3D points of shape (num_points, 3).
        trans: Transformation matrix of shape (4, 4).
    Returns:
        Transformed 3D points of shape (num_points, 3).
    """

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return trans.dot(points_h.T)[:3, :].T


def transform_3d_points_torch(
    trans: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Transforms sets of 3D points.

    Args:
        points: 3D points of shape (num_points, 3).
        trans: Transformation matrix of shape (4, 4).
    Returns:
        Transformed points of shape (batch_size, num_points, 3).
    """

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    assert trans.device == points.device
    assert trans.dtype == points.dtype
    points_h = torch.hstack(
        [points, torch.ones((points.shape[0], 1), device=points.device)]
    )
    return torch.matmul(trans, points_h.T)[:3, :].T

def gen_look_at_matrix(
    orig_camera_from_world: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
    return_camera_from_world: bool = True,
) -> np.ndarray:
    """
    Rotates the input camera such that the new transformation align the z-direction to the provided point in world.
    Args:
      camera_angle is used to apply a roll rotation around the new z
      return_camera_from_world is used to return the inverse

    Returns:
        world_from_aligned_camera or aligned_camera_from_world
    """

    center_local = transform_points(orig_camera_from_world, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_world_from_camera = np.linalg.inv(orig_camera_from_world)

    world_from_aligned_camera = orig_world_from_camera.copy()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ delta_r_local
    )

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ z_local_rot
    )

    if return_camera_from_world:
        return np.linalg.inv(world_from_aligned_camera)
    return world_from_aligned_camera

def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Transform an array of 3D points with an SE3 transform (rotation and translation).

    *WARNING* this function does not support arbitrary affine transforms that also scale
    the coordinates (i.e., if a 4x4 matrix is provided as input, the last row of the
    matrix must be `[0, 0, 0, 1]`).

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points [..., 3]

    Returns:
        Transformed points [..., 3]
    """
    return rotate_points(matrix, points) + matrix[..., :3, 3]


def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Rotates an array of 3D points with an affine transform,
    which is equivalent to transforming an array of 3D rays.

    *WARNING* This ignores the translation in `m`; to transform 3D *points*, use
    `transform_points()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points or 3d direction vectors [..., 3]

    Returns:
        Rotated points / direction vectors [..., 3]
    """
    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)

def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    # Convert the vectors to unit vectors.
    a = geometry.normalized(a_orig)
    b = geometry.normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = (
        np.eye(3, 3, dtype=a_orig.dtype)
        + v_mat
        + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))
    )

    return rot

def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res

def rotation_matrix_numpy(angle: float, direction: np.ndarray) -> np.ndarray:
    """Return a homogeneous transformation matrix [4x4] to rotate a point around the
    provided direction by a mangnitude set by angle.

    Args:
        angle: Angle to rotate around axis [rad].
        direction: Direction vector (3-vector, does not need to be normalized)

    Returns:
        M: A 4x4 matrix with the rotation component set and translation to zero.

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = geometry.normalized(direction[:3])
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    return M

def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def normalized(v: AnyTensor, axis: int = -1, eps: float = 5.43e-20) -> AnyTensor:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d
