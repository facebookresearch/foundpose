#!/usr/bin/env python3

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from utils import (
    knn_util,
    repre_util,
    template_util,
    logging, misc
)

logger: logging.Logger = logging.get_logger()

def convert_px_indices_to_im_coords(
    px_indices: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Converts pixel indices to (possibly scaled) image coordinates.

    With scale = 1.0, pixel index (i, j) corresponds to image coordinates
    (i + 0.5, j + 0.5).

    Args:
      px_indices: [n, 2] ndarray with pixel indices.
      scale: Scale to be applied to the image coordinates.
    Returns:
      [n, 2] ndarray with image coordinates.
    """

    return scale * (px_indices.float() + 0.5)

def cyclic_buddies_matching(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    query_knn_index: knn_util.KNN,
    object_features: torch.Tensor,
    object_knn_index: knn_util.KNN,
    top_k: int,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find best buddies via cyclic distance (https://arxiv.org/pdf/2204.03635.pdf)."""

    # Find nearest neighbours in both directions.
    query2obj_nn_ids = object_knn_index.search(query_features)[1].flatten()
    obj2query_nn_ids = query_knn_index.search(object_features)[1].flatten()

    # 2D locations of the query points.
    u1 = query_points

    # 2D locations of the cyclic points.
    cycle_ids = obj2query_nn_ids[query2obj_nn_ids]
    u2 = query_points[cycle_ids]

    # L2 distances between the query and cyclic points.
    cycle_dists = torch.linalg.norm(u1 - u2, axis=1)

    # Keep only top k best buddies.
    top_k = min(top_k, query_points.shape[0])
    _, query_bb_ids = torch.topk(-cycle_dists, k=top_k, sorted=True)

    # Best buddy scores.
    bb_dists = cycle_dists[query_bb_ids]
    bb_scores = torch.as_tensor(1.0 - (bb_dists / bb_dists.max()))

    # Returns IDs of the best buddies.
    object_bb_ids = query2obj_nn_ids[query_bb_ids]

    return query_bb_ids, object_bb_ids, bb_dists, bb_scores


def establish_correspondences(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    template_matching_type: str,
    feat_matching_type: str,
    top_n_templates: int,
    top_k_buddies: int,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
    template_knn_indices: Optional[List[knn_util.KNN]] = None,
    debug: bool = False,
) -> List[Dict]:
    """Establishes 2D-3D correspondences by matching image and object features."""

    timer = misc.Timer(enabled=debug)
    timer.start()
    template_ids, template_scores = template_util.template_matching(
        query_features=query_features,
        object_repre=object_repre,
        top_n_templates=top_n_templates,
        matching_type=template_matching_type,
        visual_words_knn_index=visual_words_knn_index,
    )

    timer.elapsed("Time for template matching")
    timer.start()
    # Build knn index for query features.
    query_knn_index = None
    if feat_matching_type == "cyclic_buddies":
        query_knn_index = knn_util.KNN(k=1, metric="l2")
        query_knn_index.fit(query_features)

    # Establish correspondences for each dominant template separately.
    corresps = []
    for template_counter, template_id in enumerate(template_ids):

        # Get IDs of features originating from the current template.
        tpl_feat_mask = torch.as_tensor(
            object_repre.feat_to_template_ids == template_id
        )
        tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

        # Find N best buddies.
        if feat_matching_type == "cyclic_buddies":
            assert object_repre.feat_vectors is not None
            (
                match_query_ids,
                match_obj_ids,
                match_dists,
                match_scores,
            ) = cyclic_buddies_matching(
                query_points=query_points,
                query_features=query_features,
                query_knn_index=query_knn_index,
                object_features=object_repre.feat_vectors[tpl_feat_ids],
                object_knn_index=template_knn_indices[template_id],
                top_k=top_k_buddies,
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown feature matching type ({feat_matching_type}).")

        match_obj_feat_ids = tpl_feat_ids[match_obj_ids]

        # Structures for storing 2D-3D correspondences and related info.
        coord_2d = query_points[match_query_ids]
        coord_2d_ids = match_query_ids
        assert object_repre.vertices is not None
        coord_3d = object_repre.vertices[match_obj_feat_ids]
        coord_conf = match_scores
        full_query_nn_dists = match_dists
        full_query_nn_ids = match_obj_feat_ids
        nn_vertex_ids = match_obj_feat_ids

        template_corresps = {
            "template_id": template_id,
            "template_score": template_scores[template_counter],
            "coord_2d": coord_2d,
            "coord_2d_ids": coord_2d_ids,
            "coord_3d": coord_3d,
            "coord_conf": coord_conf,
            "nn_vertex_ids": nn_vertex_ids,
        }
        # Add items for visualization/debugging.
        if debug:
            template_corresps.update(
                {
                    "nn_dists": full_query_nn_dists,
                    "nn_indices": full_query_nn_ids,
                }
            )

        corresps.append(template_corresps)

    timer.elapsed("Time for establishing corresp")

    return corresps
