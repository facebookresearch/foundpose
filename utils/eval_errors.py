#!/usr/bin/env python3

import math

from utils import misc
import numpy as np

from scipy.spatial.transform import Rotation
from bop_toolkit_lib import misc as bop_misc

def mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (mssd).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """

    pts_est = bop_misc.transform_pts_Rt(pts, R_est, t_est)
    es = []
    es_ind = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = bop_misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        err = np.linalg.norm(pts_gt_sym - pts_est, axis=1)
        es.append(err.max())
        es_ind.append(err.argmax())
    return min(es), es_ind[np.argmin(np.array(es))]


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """Maximum Symmetry-Aware Projection Distance (mspd).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """

    proj_est = bop_misc.project_pts(pts, K, R_est, t_est)
    es = []
    es_ind = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = bop_misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        err = np.linalg.norm(proj_est - proj_gt_sym, axis=1)
        es.append(err.max())
        es_ind.append(err.argmax())
    return min(es), es_ind[np.argmin(np.array(es))]


def re(R_est, R_gt) -> float: 
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def compute_rotation_error(R_est, R_gt): 
    """
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    """
    R = R_est.dot(R_gt.T)
    angles = Rotation.from_matrix(R).magnitude()

    return np.rad2deg(angles)


def compute_translation_errors(t_est, t_gt):
    """
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    """

    err = np.abs(t_est - t_gt)
    return np.linalg.norm(err, axis=-1)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):

    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_count = float(union.sum())
    if union_count > 0:
        return inter.sum() / union_count
    else:
        return 0.0

