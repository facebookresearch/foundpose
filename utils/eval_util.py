#!/usr/bin/env python3

import os

from typing import Dict, List

import numpy as np
from utils import (
    html_util,
    repre_util,
    eval_errors,
    json_util,
    logging, misc, structs, geometry
)
from utils.structs import RigidTransform, PinholePlaneCameraModel

from bop_toolkit_lib import inout

from tabulate import tabulate


logger: logging.Logger = logging.get_logger()


class EvaluatorPose:
    def __init__(self, obj_lids):

        self.obj_lids = obj_lids
        self.num_classes = len(obj_lids)
        self.class_names = [str(obj_lid) for obj_lid in obj_lids]

        self.mspd = []
        self.mssd = []
        self.mssd_n = []
        self.inliers_gt_err = []
        self.inliers_est_err = []
        self.inliers_gt = []
        self.inliers_est = []
        self.corr_dist_gt = []
        self.corr_dist_est = []
        self.point_errors = []
        self.rotation_errors = []
        self.translation_errors = []

        self.score = []
        self.R = []
        self.t = []
        self.time = []

        # Angular error of the template that is the closest to the GT orientation.
        self.template_ori_err = []

        # For the detections it is used to sort out worst cases.
        self.mask_iou = []

        # For detections (CNOS) keep the timer separately.
        self.detection_times = {}

        # keep result scene/img/obj ids
        self.result_ids = []
        self.scene_ids = []
        self.im_ids = []
        self.obj_ids = []
        self.inst_ids = []
        self.hypothesis_ids = []

        self.metrics = {
            "mspd": self.mspd,
            "mssd": self.mssd,
            "mssd_n": self.mssd_n,
        }

    def update(
        self,
        scene_id: int,
        im_id: int,
        inst_id: int,
        hypothesis_id: int,
        base_image: np.ndarray,
        object_repre_vertices: np.ndarray,
        obj_lid: int,
        object_pose_m2w: structs.ObjectPose,
        object_pose_m2w_gt: structs.ObjectPose,
        orig_camera_c2w: PinholePlaneCameraModel,
        camera_c2w: PinholePlaneCameraModel,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        corresp: Dict,
        retrieved_templates_camera_m2c: List[PinholePlaneCameraModel],
        time_per_inst: Dict,
        object_mesh_vertices: np.ndarray,
        object_syms: List[RigidTransform],
        object_diameter: float,
        inlier_radius: float = 10,
    ):

        # Transformations to the crop camera.
        trans_w2c = np.linalg.inv(camera_c2w.T_world_from_eye)
        trans_m2c_gt = trans_w2c.dot(misc.get_rigid_matrix(object_pose_m2w_gt))
        trans_m2c = trans_w2c.dot(misc.get_rigid_matrix(object_pose_m2w))

        # Transformations to the original camera.
        trans_w2oc = np.linalg.inv(orig_camera_c2w.T_world_from_eye)
        trans_m2oc_gt = trans_w2oc.dot(misc.get_rigid_matrix(object_pose_m2w_gt))
        trans_m2oc = trans_w2oc.dot(misc.get_rigid_matrix(object_pose_m2w))

        # Inlier/outliers in the GT pose.
        vertices_in_c_gt = geometry.transform_3d_points_numpy(
            trans_m2c_gt, object_repre_vertices
        )
        vertex_ids = corresp["nn_vertex_ids"]
        projs_gt = camera_c2w.eye_to_window(vertices_in_c_gt)[vertex_ids]
        corr_dist_gt = np.linalg.norm((corresp["coord_2d"] - projs_gt), axis=1)
        inliers_gt = np.where(corr_dist_gt <= inlier_radius)[0]

        # Visualize inlier/outliers in estimated pose.
        trans_m2w = misc.get_rigid_matrix(object_pose_m2w)
        trans_m2c = np.linalg.inv(camera_c2w.T_world_from_eye).dot(trans_m2w)

        # Inlier/outliers in estimated pose.
        vertices_in_c = geometry.transform_3d_points_numpy(
            trans_m2c, object_repre_vertices
        )
        projs_est = camera_c2w.eye_to_window(vertices_in_c)[vertex_ids]
        corr_dist_est = np.linalg.norm((corresp["coord_2d"] - projs_est), axis=1)
        inliers_est = np.where(corr_dist_est <= inlier_radius)[0]

        # Calculate inlier ratio for potentially many to many estimates.
        unique_2d_ids = list(dict.fromkeys(corresp["coord_2d_ids"]))
        gt_err = np.zeros(len(unique_2d_ids), dtype=float)
        est_err = np.zeros(len(unique_2d_ids), dtype=float)
        inliers_gt_err = {str(int(inlier_radius)): 0}
        inliers_est_err = {str(int(inlier_radius)): 0}
        for i in range(len(unique_2d_ids)):
            # Select the matches outsourcing from the query pixel.
            coord_2d_ids = np.where(corresp["coord_2d_ids"] == unique_2d_ids[i])[0]
            if np.sum(corr_dist_gt[coord_2d_ids] <= inlier_radius) > 0:
                gt_err[i] = 1
            if np.sum(corr_dist_est[coord_2d_ids] <= inlier_radius) > 0:
                est_err[i] = 1
            inliers_gt_err[str(int(inlier_radius))] = np.mean(gt_err)
            inliers_est_err[str(int(inlier_radius))] = np.mean(est_err)
        # old version
        # inliers_gt_err[str(thr)] = np.mean(corr_dist_gt <= thr)
        # inliers_est_err[str(thr)] = np.mean(corr_dist_est <= thr)

        score = inliers_est_err[str(int(inlier_radius))]

        R_est, t_est = trans_m2oc[:3, :3], trans_m2oc[:3, 3:]
        R_gt, t_gt = trans_m2oc_gt[:3, :3], trans_m2oc_gt[:3, 3:]
        K = misc.get_intrinsic_matrix(camera_c2w)

        # Normalize MSSD by object diameter.
        mssd_e, mssd_id = eval_errors.mssd(
            R_est, t_est, R_gt, t_gt, object_mesh_vertices, object_syms
        )
        logger.info(f"MSSD error: {mssd_e}, id: {mssd_id}")

        normalized_mssd = mssd_e / object_diameter

        # Normalize MSPD by image width.
        mspd_e, mspd_id = eval_errors.mspd(
            R_est, t_est, R_gt, t_gt, K, object_mesh_vertices, object_syms
        )
        logger.info(f"MSPD error : {mspd_e}, id: {mspd_id}")
        mspd_e = mspd_e

        # Object DPE metrics.
        point_errors = np.sqrt(np.sum((vertices_in_c_gt - vertices_in_c) ** 2, axis=-1))
        rotation_errors = eval_errors.compute_rotation_error(R_est, R_gt)
        translation_errors = eval_errors.compute_translation_errors(t_est, t_gt)
        translation_errors = np.linalg.norm(translation_errors, axis=-1)

        # Detection mask iou.
        mask_iou = eval_errors.mask_iou(pred_mask, gt_mask)

        # Calculate angular error of retrieved templates.
        R_m2c_gt = trans_m2c_gt[:3, :3]
        angular_errors = []
        for template_camera_m2c in retrieved_templates_camera_m2c:
            R_m2c_tpl = template_camera_m2c.T_world_from_eye[:3, :3]
            for sym in object_syms:
                R_m2c_gt_sym = R_m2c_gt.dot(sym["R"])
                angular_error = eval_errors.re(R_est=R_m2c_tpl, R_gt=R_m2c_gt_sym)
                angular_errors.append(angular_error)
        self.template_ori_err.append(min(angular_errors))

        self.mask_iou.append(mask_iou)

        self.mspd.append(mspd_e)
        self.mssd.append(mssd_e)
        self.mssd_n.append(normalized_mssd)
        self.inliers_gt_err.append(inliers_gt_err)
        self.inliers_est_err.append(inliers_est_err)
        self.inliers_gt.append(inliers_gt)
        self.inliers_est.append(inliers_est)
        self.corr_dist_gt.append(corr_dist_gt)
        self.corr_dist_est.append(corr_dist_est)

        self.score.append(score)
        self.R.append(R_est)
        self.t.append(t_est)
        self.time.append(time_per_inst)

        self.result_ids.append((scene_id, im_id, obj_lid, inst_id, hypothesis_id))
        self.scene_ids.append(scene_id)
        self.im_ids.append(im_id)
        self.obj_ids.append(obj_lid)
        self.inst_ids.append(inst_id)
        self.hypothesis_ids.append(hypothesis_id)

        self.point_errors.append(np.array(point_errors))
        self.rotation_errors.append(np.array(rotation_errors))
        self.translation_errors.append(np.array(translation_errors))

        return {
            "inliers_gt": inliers_gt,
            "inliers_est": inliers_est,
            "mspd": mspd_e,
            "mssd": mssd_e,
            "mspd_id": mspd_id,
            "mssd_id": mssd_id,
            "normalized_mssd": normalized_mssd,
            "inliers_gt_err": inliers_gt_err,
            "inliers_est_err": inliers_est_err,
            "corr_dist_gt": corr_dist_gt,
            "corr_dist_est": corr_dist_est,
        }

    def update_without_anno(
        self,
        scene_id: int,
        im_id: int,
        inst_id: int,
        hypothesis_id: int,
        object_repre_vertices: np.ndarray,
        obj_lid: int,
        object_pose_m2w: structs.ObjectPose,
        orig_camera_c2w: PinholePlaneCameraModel,
        camera_c2w: PinholePlaneCameraModel,
        time_per_inst: Dict,
        corresp: Dict,
        inlier_radius: float = 10,
    ):

        # Transformations to the crop camera.
        trans_w2c = np.linalg.inv(camera_c2w.T_world_from_eye)
        trans_m2c = trans_w2c.dot(misc.get_rigid_matrix(object_pose_m2w))

        # Transformations to the original camera.
        trans_w2oc = np.linalg.inv(orig_camera_c2w.T_world_from_eye)
        trans_m2oc = trans_w2oc.dot(misc.get_rigid_matrix(object_pose_m2w))

        # Visualize inlier/outliers in estimated pose.
        trans_m2w = misc.get_rigid_matrix(object_pose_m2w)
        trans_m2c = np.linalg.inv(camera_c2w.T_world_from_eye).dot(trans_m2w)
        vertices_in_c = geometry.transform_3d_points_numpy(
            trans_m2c, object_repre_vertices
        )
        projs_est = camera_c2w.eye_to_window(vertices_in_c)[corresp["nn_vertex_ids"]]
        corr_dist_est = np.linalg.norm((corresp["coord_2d"] - projs_est), axis=1)
        inliers_est = np.where(corr_dist_est <= inlier_radius)[0]

        unique_2d_ids = list(dict.fromkeys(corresp["coord_2d_ids"]))

        # Calculate inlier ratio for many to many estimates.
        est_err = np.zeros(len(unique_2d_ids), dtype=float)
        inliers_est_err = {str(int(inlier_radius)): 0}
        for i in range(len(unique_2d_ids)):
            # Select the matches outsourcing from the query pixel.
            coord_2d_ids = np.where(corresp["coord_2d_ids"] == unique_2d_ids[i])[0]
            if np.sum(corr_dist_est[coord_2d_ids] <= inlier_radius) > 0:
                est_err[i] = 1
            inliers_est_err[str(int(inlier_radius))] = np.mean(est_err)

        # Assign the score as the many to many aware inlier ratio.
        score = inliers_est_err[str(int(inlier_radius))]

        R_est, t_est = trans_m2oc[:3, :3], trans_m2oc[:3, 3:]

        self.R.append(R_est)
        self.t.append(t_est)
        self.time.append(time_per_inst)

        self.score.append(score)
        self.result_ids.append((scene_id, im_id, obj_lid, inst_id, hypothesis_id))
        self.scene_ids.append(scene_id)
        self.im_ids.append(im_id)
        self.obj_ids.append(obj_lid)
        self.inst_ids.append(inst_id)
        self.hypothesis_ids.append(hypothesis_id)

        self.inliers_est_err.append(inliers_est_err)

        return {
            "inliers_est": inliers_est,
            "inliers_est_err": inliers_est_err,
            "corr_dist_est": corr_dist_est,
        }

    def save_results_json(self, path):
        """Saves 6D object pose estimates to a file.

        :param path: Path to the output file.
        :param results: Dictionary with pose estimates.
        """
        result_info = []
        # convert everything to string.
        for i, (scene_id, img_id, obj_id, inst_id, hypothesis_id) in enumerate(
            self.result_ids
        ):

            cnos_time = self.detection_times[(scene_id, img_id)]

            if len(self.mssd) == 0:
                result_info.append(
                    {
                        "scene_id": str(scene_id),
                        "img_id": str(img_id),
                        "obj_id": str(obj_id),
                        "inst_id": str(inst_id),
                        "hypothesis_id": str(hypothesis_id),
                        "score": str(self.score[i]),
                        "R": self.R[i],
                        "t": self.t[i],
                        "time": self.time[i],
                        "cnos_time": cnos_time,
                    }
                )
            else:
                result_info.append(
                    {
                        "scene_id": str(scene_id),
                        "img_id": str(img_id),
                        "obj_id": str(obj_id),
                        "inst_id": str(inst_id),
                        "hypothesis_id": str(hypothesis_id),
                        "score": str(self.score[i]),
                        "R": self.R[i],
                        "t": self.t[i],
                        "time": self.time[i],
                        "cnos_time": cnos_time,
                        "mspd": self.mspd[i],
                        "mssd": self.mssd[i],
                        "mssd_n": self.mssd_n[i],
                        # "num_corr": len(corresp["coord_2d"]),
                        "inliers_gt": len(self.inliers_gt[i]),
                        "inliers_est": len(self.inliers_est[i]),
                        "inliers_gt_err": self.inliers_gt_err[i],
                        "inliers_est_err": self.inliers_est_err[i],
                    }
                )

        json_util.save_json(path, result_info)

    # DEPRECATED!!!
    def save_bop_results(self, path, version="bop19"):
        """Saves 6D object pose estimates to a file.

        https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#formatofresults

        :param path: Path to the output file.
        :param results: Dictionary with pose estimates.
        :param version: Version of the results.
        """
        # See docs/bop_challenge_2019.md for details.
        if version == "bop19":
            lines = ["scene_id,im_id,obj_id,score,R,t,time"]
            for i, (scene_id, img_id, obj_id, inst_id, hypothesis_id) in enumerate(
                self.result_ids
            ):
                img_select_indices = np.where(np.array(self.im_ids) == img_id)[0]
                scene_select_indices = np.where(np.array(self.scene_ids) == scene_id)[0]
                select_indices = np.intersect1d(
                    img_select_indices, scene_select_indices
                )
                run_time = np.array(self.time)[select_indices].sum()
                if len(self.detection_times.keys()) != 0:
                    run_time = run_time + self.detection_times[(scene_id, img_id)]

                lines.append(
                    "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                        scene_id=scene_id,
                        im_id=img_id,
                        obj_id=obj_id,
                        score=self.score[i],
                        R=" ".join(map(str, self.R[i].flatten().tolist())),
                        t=" ".join(map(str, self.t[i].flatten().tolist())),
                        time=run_time,
                    )
                )

            with os.open(path, "wb") as f:
                f.write("\n".join(lines).encode("utf-8"))

        else:
            raise ValueError("Unknown version of BOP results.")

    def save_metrics(self, path, inlier_thresh):

        inlier_thresh_key = str(int(inlier_thresh))

        mssd_per_class = []
        mssd_n_per_class = []
        mspd_per_class = []
        num_objects = []
        inliers_gt_per_class = []
        inliers_est_per_class = []

        point_err_p50_per_class = []
        point_err_p95_per_class = []
        rot_err_p50_per_class = []
        rot_err_p95_per_class = []
        trans_err_p50_per_class = []
        trans_err_p95_per_class = []
        tpl_ori_err_per_class = []

        point_err_p50 = np.percentile(np.concatenate(self.point_errors, axis=0), 50)
        point_err_p95 = np.percentile(np.concatenate(self.point_errors, axis=0), 95)
        rot_err_p50 = np.percentile(self.rotation_errors, 50)
        rot_err_p95 = np.percentile(self.rotation_errors, 95)
        trans_err_p50 = np.percentile(self.translation_errors, 50)
        trans_err_p95 = np.percentile(self.translation_errors, 95)
        tpl_ori_err = np.mean(self.template_ori_err)

        for obj_lid in self.obj_lids:
            selected_ids = np.where(np.array(self.obj_ids) == obj_lid)[0]
            mssd_per_class.append(np.nanmean(np.array(self.mssd)[selected_ids]))
            mssd_n_per_class.append(np.nanmean(np.array(self.mssd_n)[selected_ids]))
            mspd_per_class.append(np.nanmean(np.array(self.mspd)[selected_ids]))
            num_objects.append(len(selected_ids))
            inliers_gt_per_class.append(
                np.nanmean(
                    [self.inliers_gt_err[id][inlier_thresh_key] for id in selected_ids]
                )
            )
            inliers_est_per_class.append(
                np.nanmean(
                    [self.inliers_est_err[id][inlier_thresh_key] for id in selected_ids]
                )
            )
            point_err_p50_per_class.append(
                np.nanpercentile(
                    np.array(list(np.array(self.point_errors)[selected_ids])), 50
                )
            )
            point_err_p95_per_class.append(
                np.nanpercentile(
                    np.array(list(np.array(self.point_errors)[selected_ids])), 95
                )
            )
            rot_err_p50_per_class.append(
                np.nanpercentile(np.array(self.rotation_errors)[selected_ids], 50)
            )
            rot_err_p95_per_class.append(
                np.nanpercentile(np.array(self.rotation_errors)[selected_ids], 95)
            )
            trans_err_p50_per_class.append(
                np.nanpercentile(np.array(self.translation_errors)[selected_ids], 50)
            )
            trans_err_p95_per_class.append(
                np.nanpercentile(np.array(self.translation_errors)[selected_ids], 95)
            )
            tpl_ori_err_per_class.append(
                np.nanmean(np.array(self.template_ori_err)[selected_ids])
            )

        header = ["", "overall", "pmean", "sym", "nonsym"] + self.class_names

        table = [
            ["mssd", np.nanmean(self.mssd), np.nanmean(mssd_per_class), 0, 0]
            + mssd_per_class,
            ["mssd_n", np.nanmean(self.mssd_n), np.nanmean(mssd_n_per_class), 0, 0]
            + mssd_n_per_class,
            ["mspd", np.nanmean(self.mspd), np.nanmean(mspd_per_class), 0, 0]
            + mspd_per_class,
            [
                "inliers_gt",
                np.nanmean([err[inlier_thresh_key] for err in self.inliers_gt_err]),
                np.nanmean(inliers_gt_per_class),
                0,
                0,
            ]
            + inliers_gt_per_class,
            [
                "inliers_est",
                np.nanmean([err[inlier_thresh_key] for err in self.inliers_est_err]),
                np.nanmean(inliers_est_per_class),
                0,
                0,
            ]
            + inliers_est_per_class,
            ["Point_p50", point_err_p50, 0, 0, 0] + point_err_p50_per_class,
            ["Point_p95", point_err_p95, 0, 0, 0] + point_err_p95_per_class,
            ["Rot_p50", rot_err_p50, 0, 0, 0] + rot_err_p50_per_class,
            ["Rot_p95", rot_err_p95, 0, 0, 0] + rot_err_p95_per_class,
            ["Trans_p50", trans_err_p50, 0, 0, 0] + trans_err_p50_per_class,
            ["Trans_p95", trans_err_p95, 0, 0, 0] + trans_err_p95_per_class,
            ["Tpl_ori_err", tpl_ori_err, 0, 0, 0] + tpl_ori_err_per_class,
            ["num_obj", np.sum(num_objects), np.mean(num_objects), 0, 0] + num_objects,
        ]

        with os.open(path, "wb") as f:
            # f.write("\n".join(lines).encode("utf-8"))
            # In order to unify format, remove all the alignments.
            f.write(
                tabulate(
                    table,
                    headers=header,
                    tablefmt="tsv",
                    floatfmt=".2f",
                    numalign=None,
                    stralign=None,
                ).encode("utf-8")
            )

    def top_n(self, output_dir, n=100, metric_key="mspd", im_ext=".jpg"):
        # Get the top n best and worst predictions according to a metric.
        # This function reads pre-saved images and combines them in a single file.
        # Write the final file as an html.
        score_ids = np.argsort(self.metrics[metric_key])

        # best_n = score_ids[:n]
        # worst_n = score_ids[-n:][::-1]
        best_n = score_ids
        worst_n = score_ids[::-1]

        best_iou = np.array(self.mask_iou)[best_n]
        worst_iou = np.array(self.mask_iou)[worst_n]

        image_content = ""
        num_selected = 0
        for res_i, (scene_id, img_id, obj_id, inst_id, hypothesis_id) in enumerate(
            np.array(self.result_ids)[best_n]
        ):
            if best_iou[res_i] <= 0.2:
                continue
            if num_selected > n:
                break
            vis_path = os.join(
                output_dir,
                f"{scene_id}_{img_id}_{obj_id}_{inst_id}_{hypothesis_id}{im_ext}",
            )
            res_image = inout.load_image(vis_path)
            # read previously saved image, add to html page with a header
            image_content += html_util.add_text(
                f"Scene: {scene_id} Image: {img_id}, Obj id:{obj_id}, Inst id:{inst_id}, Hypothesis id:{hypothesis_id}"
            )
            image_content += html_util.add_rgb(
                res_image,
                f"{scene_id}_{img_id}_{obj_id}_{inst_id}_{hypothesis_id}{im_ext}",
            )
            image_content = html_util.linebreakHTML(image_content)
            num_selected += 1

        results_path = os.join(output_dir, "best_n.html")
        html_str = html_util.wrapHTMLBody(image_content)
        with os.open(results_path, "wb") as f:
            f.write(html_str.encode("utf-8"))

        image_content = ""
        num_selected = 0
        for res_i, (scene_id, img_id, obj_id, inst_id, hypothesis_id) in enumerate(
            np.array(self.result_ids)[worst_n]
        ):
            if worst_iou[res_i] <= 0.2:
                continue
            if num_selected > n:
                break
            vis_path = os.join(
                output_dir,
                f"{scene_id}_{img_id}_{obj_id}_{inst_id}_{hypothesis_id}{im_ext}",
            )
            res_image = inout.load_image(vis_path)
            # read previously saved image, add to html page with a header
            image_content += html_util.add_text(
                f"Scene: {scene_id} Image: {img_id}, Obj id:{obj_id}, Inst id:{inst_id}"
            )
            image_content += html_util.add_rgb(
                res_image,
                f"{scene_id}_{img_id}_{obj_id}_{inst_id}_{hypothesis_id}{im_ext}",
            )
            image_content = html_util.linebreakHTML(image_content)
            num_selected += 1

        results_path = os.join(output_dir, "worst_n.html")
        html_str = html_util.wrapHTMLBody(image_content)
        with os.open(results_path, "wb") as f:
            f.write(html_str.encode("utf-8"))
