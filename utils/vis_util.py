#!/usr/bin/env python3

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from utils import (
    projector_util,
    repre_util,
    vis_base_util,
    renderer_base,
    render_vis_util,
    logging, 
    structs, 
    misc, 
    geometry
)

from utils.misc import tensor_to_array, array_to_tensor

logger: logging.Logger = logging.get_logger()


def vis_color_point_cloud(
    base_image: np.ndarray,
    camera: structs.CameraModel,
    points_in_c: np.ndarray,
    colors: np.ndarray,
    dpi: int = 100,
) -> np.ndarray:

    # Sort the projections for drawing from the furthest to the closest.
    order = np.argsort(points_in_c[:, 2])[::-1]

    points_in_c = points_in_c[order]
    colors = np.array(colors)[order]

    # Project the vertices to the image plane.
    projs = camera.eye_to_window(points_in_c)

    depths = points_in_c[:, 2][order]
    depths_min = depths.min()
    depths_max = depths.max()

    point_sizes = (depths - depths_min) / (depths_max - depths_min)
    point_sizes = 1.0 - point_sizes
    point_sizes = 0.03 * dpi * (1.0 + point_sizes)

    point_sizes = 8 * np.ones_like(depths)

    # Filter the out of image points to prevent visualization.
    mask = np.logical_and(
        np.logical_and(
            0 + point_sizes <= projs[:, 0],
            projs[:, 0] < base_image.shape[1] - point_sizes,
        ),
        np.logical_and(
            0 + point_sizes <= projs[:, 1],
            projs[:, 1] < base_image.shape[0] - point_sizes,
        ),
    )
    projs = projs[mask]
    colors = colors[mask]

    # base_image_vis = (0.3 * base_image).astype(np.uint8)
    base_image_vis = (0.5 * base_image).astype(np.uint8)
    vis_base_util.plot_images(imgs=[base_image_vis], dpi=dpi)
    vis_base_util.plot_keypoints(kpts=[projs], colors=[colors], ps=[point_sizes])

    return vis_base_util.save_plot_to_ndarray()

def vis_pointcloud_error(
    object_repre: repre_util.FeatureBasedObjectRepre,
    object_pose_m2w: structs.ObjectPose,
    object_pose_m2w_gt: structs.ObjectPose,
    camera_c2w: structs.CameraModel,
    mssd_id: int,
    ply_output_path: str,
):

    # Visualize the object representation.
    # object pose from world to camera:
    trans_m2w_gt = misc.get_rigid_matrix(object_pose_m2w_gt)
    trans_m2c_gt = np.linalg.inv(camera_c2w.T_world_from_eye).dot(trans_m2w_gt)
    vertices_in_c_gt = geometry.transform_3d_points_numpy(
        trans_m2c_gt, object_repre.vertices
    )
    # projs_gt = camera_c2w.eye_to_window(vertices_in_c_gt)

    trans_m2w = misc.get_rigid_matrix(object_pose_m2w)
    trans_m2c = np.linalg.inv(camera_c2w.T_world_from_eye).dot(trans_m2w)
    vertices_in_c = geometry.transform_3d_points_numpy(
        trans_m2c, object_repre.vertices
    )
    # projs_estimate = camera_c2w.eye_to_window(vertices_in_c)

    vertices = np.concatenate((vertices_in_c_gt, vertices_in_c), axis=0)

    vertex_colors_gt = np.repeat(
        np.array([[0.0, 1.0, 0.0]]), vertices_in_c_gt.shape[0], axis=0
    )
    # vertex_colors_gt[mssd_id] = np.array([[1.0, 0.0, 0.0]])

    vertex_colors_est = np.repeat(
        np.array([[0.0, 0.0, 1.0]]), vertices_in_c_gt.shape[0], axis=0
    )
    # vertex_colors_est[mssd_id] = np.array([[1.0, 0.0, 0.0]])

    vertex_colors = np.concatenate((vertex_colors_gt, vertex_colors_est), axis=0)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        vertex_colors=vertex_colors,
        process=False,
    )
    
    with open(ply_output_path, "wb") as f:
        f.write(trimesh.exchange.ply.export_ply(mesh))


def vis_pca_feature_map(
    feature_map_chw: np.ndarray,
    image_height: int,
    image_width: int,
    pca_projector: projector_util.PCAProjector,  # object_repre.projector_vis
):

    # PCA visualization.
    feature_map_chw_up = F.interpolate(
        feature_map_chw.unsqueeze(0),
        size=(image_height, image_width),
        mode="nearest",
    )[0]
    feature_map_hwc_np = misc.chw_to_hwc(tensor_to_array(feature_map_chw_up))

    vis_pca_components = min(pca_projector.pca.n_components, 3)  # 6)
    map_width = feature_map_hwc_np.shape[1]
    map_height = feature_map_hwc_np.shape[0]
    query_pca_transform = pca_projector.pca.transform(
        feature_map_hwc_np.reshape(map_width * map_height, -1)
    )
    query_pca_feature_map = query_pca_transform.reshape((map_height, map_width, -1))

    vis_pca_features = None
    for i in range(vis_pca_components // 3):
        vis_pca_features_each = (
            255
            * vis_base_util.normalize_data(
                query_pca_feature_map[:, :, i * 3 : (i + 1) * 3]
            )
        ).astype(np.uint8)
        if i == 0:
            vis_pca_features = vis_pca_features_each

    # Make sure the feature map is of the expected size.
    vis_pca_features = misc.resize_image(
        image=vis_pca_features,
        size=(image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return vis_pca_features


def set_bg_to_gray(im, bg_thresh, gray_level):
    bg_mask = np.mean(im.astype(np.float32), axis=2) < bg_thresh
    kernel = np.ones((3, 3), np.uint8)
    bg_mask = cv2.dilate(bg_mask.astype(np.uint8), kernel, iterations=1)
    im[bg_mask.astype(bool)] = gray_level
    return im


def vis_inference_results(
    base_image: np.ndarray,
    object_repre: repre_util.FeatureBasedObjectRepre,
    object_lid: int,
    object_pose_m2w: structs.ObjectPose,
    object_pose_m2w_gt: Optional[structs.ObjectPose],
    feature_map_chw: torch.Tensor,
    feature_map_chw_proj: torch.Tensor,
    vis_feat_map: bool,
    object_box: np.ndarray,
    object_mask: np.ndarray,
    camera_c2w: structs.CameraModel,
    corresp: Dict,
    matched_template_ids: List[int],
    matched_template_scores: List[float],
    best_template_ind: int,
    renderer: renderer_base.RendererBase,
    pose_eval_dict: Dict,
    corresp_top_n: int = 50,
    dpi: int = 100,
    inlier_thresh: float = 10,
    object_pose_m2w_coarse: Optional[structs.ObjectPose] = None,
    pose_eval_dict_coarse: Optional[Dict] = None,
    vis_for_paper: bool = True,
    vis_for_teaser: bool = False,
    extractor: Any = None,
):

    device = feature_map_chw.device

    accent_color = (255, 255, 255)
    # accent_color = (0, 255, 0)
    if vis_for_paper:
        vis_margin = 0
        match_alpha = 0.8
    elif vis_for_teaser:
        vis_margin = 8
        match_alpha = 0.8
    else:
        vis_margin = 0
        match_alpha = 0.5

    # For visualization for paper.
    paper_gray = 230
    paper_bg_thresh = 5

    image_height = base_image.shape[0]
    image_width = base_image.shape[1]

    # if vis_feat_map:
    #     base_image_vis = vis_pca_feature_map(
    #         feature_map_chw=feature_map_chw_proj,
    #         image_height=image_height,
    #         image_width=image_width,
    #         pca_projector=object_repre.projector_vis,
    #     )
    # else:
    base_image_vis = (0.4 * base_image).astype(np.uint8)

    vis_tiles = []

    template_id = int(matched_template_ids[best_template_ind])
    template = object_repre.templates[template_id]

    query_feat_vis = None
    template_feat_vis = None
    if vis_feat_map:
        template_tensor_chw = array_to_tensor(template).to(torch.float32)/255.0
        template_tensor_bchw = template_tensor_chw.unsqueeze(0).to(device)
        extractor_output = extractor(template_tensor_bchw)
        template_feature_map_chw = extractor_output["feature_maps"][0]

        # channels = feature_map_chw.shape[0]
        # features_for_pca = torch.vstack(
        #     [
        #         feature_map_chw.permute(1, 2, 0).reshape((-1, channels)),
        #         template_feature_map_chw.permute(1, 2, 0).reshape((-1, channels)),
        #     ]
        # )

        # pca_projector_vis = projector_util.PCAProjector(dim=3, whiten=False)
        # pca_projector_vis.fit(features_for_pca)

        query_feat_vis = vis_pca_feature_map(
            feature_map_chw=feature_map_chw,
            image_height=image_height,
            image_width=image_width,
            pca_projector=object_repre.feat_vis_projectors[0],
        )
        template_feat_vis = vis_pca_feature_map(
            feature_map_chw=template_feature_map_chw,
            image_height=image_height,
            image_width=image_width,
            pca_projector=object_repre.feat_vis_projectors[0],
        )

    # ------------------------------------------------------------------------------
    # Row 1: Query image with object poses
    # ------------------------------------------------------------------------------

    if not vis_for_teaser:

        # Row 1 left

        # Show object mask.
        mask_3c = np.tile(np.expand_dims(object_mask.astype(np.float32), -1), (1, 1, 3))
        mask_3c_bool = mask_3c.astype(bool)
        if vis_for_paper:
            vis = np.array(base_image).astype(np.float32)
            vis[mask_3c_bool] *= 0.5
            vis[mask_3c_bool] += np.tile(
                0.5 * np.array(accent_color), (int(np.sum(object_mask)), 1)
            ).flatten()
            vis = vis.astype(np.uint8)
        else:
            vis = np.array(base_image_vis)
            vis[mask_3c_bool] = 0.5 * vis[mask_3c_bool] + 0.5 * 255
            vis = vis.astype(np.uint8)

        vis_base_util.plot_images(imgs=[vis], dpi=dpi)
        if not vis_for_paper:
            vis_base_util.add_text(0, "Input mask")
        # vis_base_util.plot_boundingbox(object_box)
        vis_tile_left = vis_base_util.save_plot_to_ndarray()

        # ROW 1 right

        vis = np.array(base_image)

        # Show contours of the object in the GT pose.
        if object_pose_m2w_gt is not None:
            # vis_gt_pose = render_vis_util.vis_posed_meshes_of_objects(
            #     base_image=np.ones_like(base_image) * 255,
            #     object_lids=[object_lid],
            #     object_poses_m2w=[object_pose_m2w_gt],
            #     camera_c2w=camera_c2w,
            #     renderer=renderer,
            #     object_colors=[(0.0, 0.0, 0.0)],
            #     object_stickers=None,
            #     fg_opacity=1.0,
            #     bg_opacity=1.0,
            #     all_in_one=True,
            # )[0]
            vis_gt_pose = render_vis_util.create_object_mask(
                base_image=np.ones_like(base_image) * 255,
                object_lids=[object_lid],
                object_poses_m2w=[object_pose_m2w_gt],
                camera_c2w=camera_c2w,
                renderer=renderer,
                object_colors=[(0.0, 0.0, 0.0)],
                object_stickers=None,
                fg_opacity=1.0,
                bg_opacity=1.0,
                all_in_one=True,
            )
            vis = vis_base_util.add_contour_overlay(
                vis,
                vis_gt_pose,
                color=(255, 0, 0),
                dilate_iterations=1,
            )

        # Show contours of the object in the coarse estimated pose.
        if object_pose_m2w_coarse is not None:
            vis_coarse_est_pose = render_vis_util.vis_posed_meshes_of_objects(
                base_image=np.ones_like(base_image) * 255,
                object_lids=[object_lid],
                object_poses_m2w=[object_pose_m2w_coarse],
                camera_c2w=camera_c2w,
                renderer=renderer,
                object_colors=[(0.0, 0.0, 0.0)],
                object_stickers=None,
                fg_opacity=1.0,
                bg_opacity=1.0,
                all_in_one=True,
            )[0]
            vis = vis_base_util.add_contour_overlay(
                vis,
                vis_coarse_est_pose,
                color=(0, 0, 255),
                dilate_iterations=1,
            )

        # Show contours of the object in the final estimated pose.
        # vis_est_pose = render_vis_util.vis_posed_meshes_of_objects(
        #     base_image=np.ones_like(base_image) * 255,
        #     object_lids=[object_lid],
        #     object_poses_m2w=[object_pose_m2w],
        #     camera_c2w=camera_c2w,
        #     renderer=renderer,
        #     object_colors=[(0.0, 0.0, 0.0)],
        #     object_stickers=None,
        #     fg_opacity=1.0,
        #     bg_opacity=1.0,
        #     all_in_one=True,
        # )[0]
        vis_est_pose = render_vis_util.create_object_mask(
            base_image=np.ones_like(base_image) * 255,
            object_lids=[object_lid],
            object_poses_m2w=[object_pose_m2w],
            camera_c2w=camera_c2w,
            renderer=renderer,
            object_colors=[(0.0, 0.0, 0.0)],
            object_stickers=None,
            fg_opacity=1.0,
            bg_opacity=1.0,
            all_in_one=True,
        )
        vis = vis_base_util.add_contour_overlay(
            vis,
            vis_est_pose,
            color=(0, 255, 0),
            dilate_iterations=1,
        )

        vis_base_util.plot_images(imgs=[vis], dpi=dpi)

        if not vis_for_paper:
            txt = "Poses (R: GT, B: coarse, G: final)"
            if "mssd" in pose_eval_dict:
                mssd = pose_eval_dict["mssd"]
                mssd_n = pose_eval_dict["normalized_mssd"]
                txt += f"\nMSSD: {mssd:.2f} ({mssd_n:.2f})"
            if "mspd" in pose_eval_dict:
                mspd = pose_eval_dict["mspd"]
                txt += f"\nMSPD: {mspd:.2f}"
            vis_base_util.add_text(0, txt)

        else:
            vis_tile_left = np.hstack(
                [
                    vis_tile_left,
                    255 * np.ones((vis_tile_left.shape[0], vis_margin, 3), np.uint8),
                ]
            )

        # Finalize row 1
        vis_tile_right = vis_base_util.save_plot_to_ndarray()
        tile = np.hstack([vis_tile_left, vis_tile_right])
        vis_tiles.append(tile)
        if vis_for_paper:
            vis_tiles.append(255 * np.ones((vis_margin, tile.shape[1], 3), np.uint8))

    # ------------------------------------------------------------------------------
    # ROW 2: Matched templates
    # ------------------------------------------------------------------------------

    if not vis_for_teaser:

        matched_templates = [
            np.asarray(object_repre.templates[i].astype(np.float32)/255.0)
            for i in matched_template_ids
        ]
        matched_templates = [np.transpose(t, (1, 2, 0)) for t in matched_templates]
        tpls_tile = np.hstack(matched_templates)
        tpls_tile_size = (
            2 * image_width + vis_margin,
            int(tpls_tile.shape[0] * 2 * image_width / tpls_tile.shape[1]),
        )

        # if vis_for_paper:
        # tpls_tile = set_bg_to_gray(tpls_tile, paper_bg_thresh, paper_gray)

        tpls_tile = misc.resize_image(
            image=tpls_tile,
            size=tpls_tile_size,
            interpolation=cv2.INTER_AREA,
        )

        vis_base_util.plot_images(imgs=[tpls_tile], dpi=dpi)

        if not vis_for_paper:
            tpls_ids_str = ""
            tpls_scores_str = ""
            for tpl_id in range(len(matched_template_ids)):
                tpls_ids_str += f"{matched_template_ids[tpl_id]}"
                if tpl_id == best_template_ind:
                    tpls_ids_str += "*"
                tpls_scores_str += f"{matched_template_scores[tpl_id]:.2f}"
                if tpl_id != len(matched_template_ids) - 1:
                    tpls_ids_str += ", "
                    tpls_scores_str += ", "
            vis_base_util.add_text(
                0, f"Matched tpls: {tpls_ids_str}\nScores: {tpls_scores_str}"
            )

        tile = vis_base_util.save_plot_to_ndarray()
        vis_tiles.append(tile)
        if vis_for_paper:
            vis_tiles.append(255 * np.ones((vis_margin, tile.shape[1], 3), np.uint8))

    # ------------------------------------------------------------------------------
    # ROW 3: Matches
    # ------------------------------------------------------------------------------

    # Visualize the matches from the best template.

    # Get left 2D points.
    selected_ids = corresp["coord_conf"].argsort()[::-1][:corresp_top_n]
    kpts_left = corresp["coord_2d"][selected_ids]

    # Get right 2D points.
    tpl_cameras_m2c = object_repre.template_cameras_cam_from_model[template_id]
    all_tpl_vertex_ids = corresp["nn_vertex_ids"]
    all_tpl_vertices_in_c = geometry.transform_3d_points_numpy(
        np.linalg.inv(tpl_cameras_m2c.T_world_from_eye), object_repre.vertices[all_tpl_vertex_ids]
    )
    all_kpts_right = tpl_cameras_m2c.eye_to_window(all_tpl_vertices_in_c)
    kpts_right = all_kpts_right[selected_ids]

    # Returns colors from blue (most confident) to red (least confident) in turbo map.
    right_image = np.asarray(255 * template, np.uint8)
    match_offset = np.array([0, 0])
    if vis_for_paper:

        # if vis_for_teaser:
        left_image = np.asarray(0.9 * query_feat_vis.astype(np.float32), np.uint8)
        intensity_factor = 1.0
        right_image = np.asarray(
            intensity_factor * template_feat_vis.astype(np.float32), np.uint8
        )
        match_color = (230, 230, 230)
        match_colors = tuple(np.array(match_color) / 255.0)
        # match_alphas = corresp["coord_conf"][selected_ids][::-1]
        match_alphas = [1.0 for _ in range(len(selected_ids))]
        # else:
        #     left_image = base_image
        #     right_image = set_bg_to_gray(right_image, paper_bg_thresh, paper_gray)
        #     match_colors = tuple(np.array(accent_color) / 255.0)
        #     match_alphas = corresp["coord_conf"][selected_ids][::-1]

        # Add border.
        vis_margin_half = int(0.5 * vis_margin)
        border = 255 * np.ones((left_image.shape[0], vis_margin_half, 3), np.uint8)
        left_image = np.hstack([left_image, border])
        right_image = np.hstack([border, right_image])
        match_offset = np.array([vis_margin_half, 0])

        # Make bg black.
        # template_small = misc.resize_image(
        #     template, (template.shape[1] // 14, template.shape[0] // 14)
        # )
        # bg_mask = 255 * np.mean(template_small.astype(np.float32), axis=2) < 5
        # bg_mask = misc.resize_image(
        #     bg_mask.astype(np.uint8),
        #     (template.shape[1], template.shape[0]),
        #     interpolation=cv2.INTER_NEAREST,
        # ).astype(bool)
        # right_image[bg_mask] = 0

    else:
        match_colors = vis_base_util.get_colormap(len(selected_ids))[::-1]
        match_alphas = [0.5 for _ in range(len(selected_ids))]
        left_image = base_image_vis
        right_image = np.transpose(right_image, (1,2,0))

    vis_base_util.plot_images(imgs=[left_image, right_image], dpi=dpi)

    # Draw the less similar lines before the more similars to see better.
    a = np.linspace(0.2, 1, len(selected_ids)).tolist()
    vis_base_util.plot_matches(
        kpts_left[::-1],
        kpts_right[::-1] + match_offset,
        color=match_colors,
        lw=1.0,
        ps=5,
        a=list(match_alphas),
        w=image_width,
        h=image_height,
    )

    if not vis_for_paper:
        p5, p10, p20, p40, p80 = (
            np.percentile(corresp["nn_dists"], 5),
            np.percentile(corresp["nn_dists"], 10),
            np.percentile(corresp["nn_dists"], 20),
            np.percentile(corresp["nn_dists"], 40),
            np.percentile(corresp["nn_dists"], 80),
        )
        corresp_count = len(corresp["coord_2d"])
        vis_base_util.add_text(
            0,
            f"Corresp count: {corresp_count}\nDist (p5|10|20|40|80):\n{p5:.2f}, {p10:.2f}, {p20:.2f}, {p40:.2f}, {p80:.2f}",
        )

    tile = vis_base_util.save_plot_to_ndarray()
    vis_tiles.append(tile)

    if vis_for_teaser:
        left_image = base_image
        right_image = np.asarray(255 * template, np.uint8)

        # Add border.
        vis_margin_half = int(0.5 * vis_margin)
        border = 255 * np.ones((left_image.shape[0], vis_margin_half, 3), np.uint8)
        left_image = np.hstack([left_image, border])
        right_image = np.hstack([border, right_image])

        vis_base_util.plot_images(imgs=[left_image, right_image], dpi=dpi)

        tile = vis_base_util.save_plot_to_ndarray()
        vis_tiles.append(tile)

    # ------------------------------------------------------------------------------
    # ROW 3b: Feature maps
    # ------------------------------------------------------------------------------

    # if vis_for_paper:
    #     # Add border.
    #     vis_margin_half = int(0.5 * vis_margin)
    #     border = 255 * np.ones((left_image.shape[0], vis_margin_half, 3), np.uint8)
    #     query_feat_vis = np.hstack([query_feat_vis, border])
    #     template_feat_vis = np.hstack([border, template_feat_vis])

    # vis_base_util.plot_images(imgs=[query_feat_vis, template_feat_vis], dpi=dpi)
    # tile = vis_base_util.save_plot_to_ndarray()
    # vis_tiles.append(tile)

    # ------------------------------------------------------------------------------
    # ROW 4: Visualize inlier matches w.r.t. estimated pose
    # ------------------------------------------------------------------------------

    if not vis_for_paper:
        inliers_est = pose_eval_dict["inliers_est"]
        inliers_est_err = pose_eval_dict["inliers_est_err"]

        selected_ids = corresp["coord_conf"][inliers_est].argsort()[::-1][
            :corresp_top_n
        ]
        selected_inliers_est = np.array(inliers_est[selected_ids])
        #vis_base_util.plot_images(imgs=[base_image_vis, template], dpi=dpi)
        vis_base_util.plot_images(imgs=[base_image_vis, right_image], dpi=dpi)

        if len(selected_inliers_est) != 0:
            vis_base_util.plot_matches(
                corresp["coord_2d"][selected_inliers_est],
                all_kpts_right[selected_inliers_est],
                color=(0.0, 1.0, 0.0),
                lw=1.0,
                ps=2,
                a=0.5,
                w=image_width,
                h=image_height,
            )

        corr_dist_est = pose_eval_dict["corr_dist_est"]
        p5, p10, p20, p40, p80 = (
            np.percentile(corr_dist_est, 5),
            np.percentile(corr_dist_est, 10),
            np.percentile(corr_dist_est, 20),
            np.percentile(corr_dist_est, 40),
            np.percentile(corr_dist_est, 80),
        )
        inliers_ratio = inliers_est_err[str(int(inlier_thresh))]
        vis_base_util.add_text(
            0,
            f"Inliers w.r.t. estimated pose\n"
            f"Err (p5|10|20|40|80):\n"
            f"{p5:.2f}, {p10:.2f}, {p20:.2f}, {p40:.2f}, {p80:.2f}\n"
            f"Inlier ratio: {inliers_ratio:.2f} (th: {inlier_thresh:.1f}px)",
        )

        vis_tiles.append(vis_base_util.save_plot_to_ndarray())

    # ------------------------------------------------------------------------------
    # ROW 5: Visualize inlier matches w.r.t. GT pose
    # ------------------------------------------------------------------------------

    if False and not vis_for_paper and object_pose_m2w_gt is not None:
        inliers_gt = pose_eval_dict["inliers_gt"]
        inliers_gt_err = pose_eval_dict["inliers_gt_err"]

        selected_ids = corresp["coord_conf"][inliers_gt].argsort()[::-1][:corresp_top_n]
        selected_inliers_gt = np.array(inliers_gt[selected_ids])

        vis_base_util.plot_images(imgs=[base_image_vis, template], dpi=dpi)

        if len(selected_inliers_gt) != 0:
            vis_base_util.plot_matches(
                corresp["coord_2d"][selected_inliers_gt],
                all_kpts_right[selected_inliers_gt],
                color=(0.0, 1.0, 0.0),
                lw=1.0,
                ps=2,
                a=0.5,
                w=image_width,
                h=image_height,
            )

        corr_dist_gt = pose_eval_dict["corr_dist_gt"]
        p5, p10, p20, p40, p80 = (
            np.percentile(corr_dist_gt, 5),
            np.percentile(corr_dist_gt, 10),
            np.percentile(corr_dist_gt, 20),
            np.percentile(corr_dist_gt, 40),
            np.percentile(corr_dist_gt, 80),
        )
        inliers_ratio = inliers_gt_err[str(int(inlier_thresh))]
        vis_base_util.add_text(
            0,
            f"Inliers w.r.t. GT pose\n"
            f"Err (p5|10|20|40|80):\n"
            f"{p5:.2f}, {p10:.2f}, {p20:.2f}, {p40:.2f}, {p80:.2f}\n"
            f"Inlier ratio: {inliers_ratio:.2f} (th: {inlier_thresh:.1f}px)",
        )

        vis_tiles.append(vis_base_util.save_plot_to_ndarray())

    return vis_tiles

