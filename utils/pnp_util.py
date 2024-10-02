#!/usr/bin/env python3

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from utils import (
    logging,
    misc
)
from utils.misc import tensor_to_array

from utils.structs import AlignedBox2f, PinholePlaneCameraModel

logger: logging.Logger = logging.get_logger()


def estimate_pose(
    corresp: Dict[str, Any],
    camera_c2w: PinholePlaneCameraModel,
    pnp_type: str,
    pnp_ransac_iter: int,
    pnp_inlier_thresh: float,
    pnp_required_ransac_conf: float,
    pnp_refine_lm: bool,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimates pose from provided 2D-3D correspondences and camera intrinsics.

    Args:
        corresp: correspondence dictionary as returned by corresp_util. Has the following:
            - coord_2d (num_points, 2): pixel coordinates from query image
            - coord_3d (num_points, 3): point coordinates from the 3d object representation
            - nn_distances (num_points) : cosine distances as returned by KNN
            - nn_indices (num_points).: indices within the object representations
        camera_c2w: camera intrinsics.
    """

    if pnp_type == "opencv":

        object_points = tensor_to_array(corresp["coord_3d"]).astype(np.float32)
        image_points = tensor_to_array(corresp["coord_2d"]).astype(np.float32)
        K = misc.get_intrinsic_matrix(camera_c2w)
        try:
            pose_est_success, rvec_est_m2c, t_est_m2c, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=pnp_ransac_iter,
                reprojectionError=pnp_inlier_thresh,
                confidence=pnp_required_ransac_conf,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            # Added to avoid a crash in cv2.solvePnPRansac due to too less correspondences
            # (even though more than 6 are provided, some of them may be colinear...).
            pose_est_success = False
            r_est_m2c = None
            t_est_m2c = None
            inliers = None
            quality = None
        else:
            # Optional LM refinement on inliers.
            if pose_est_success and pnp_refine_lm:
                rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                    objectPoints=object_points[inliers],
                    imagePoints=image_points[inliers],
                    cameraMatrix=K,
                    distCoeffs=None,
                    rvec=rvec_est_m2c,
                    tvec=t_est_m2c,
                )

            r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]
            quality = 0.0
            if pose_est_success:
                quality = float(len(inliers))

    elif pnp_type is None:
        raise ValueError("Unsupported PnP type")

    return pose_est_success, r_est_m2c, t_est_m2c, inliers, quality
