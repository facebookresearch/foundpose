#!/usr/bin/env python3

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2

import numpy as np

from utils import (
    eval_errors,
    config_util,
    json_util,
    misc,
    logging
)

from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

logger = logging.get_logger()


def load_detections_in_bop_format(path: str):
    """Load detections saved in the BOP format."""

    pred_mask_list = json_util.load_json(path)
    detections = defaultdict(list)

    for pred in pred_mask_list:
        key = (pred["scene_id"], pred["image_id"], pred["category_id"])
        detections[key].append(
            {
                "bbox": pred["bbox"],
                "segmentation": pred["segmentation"],
                "score": pred["score"],
                "time": pred["time"],
            }
        )

    return detections


def get_instances_for_pose_estimation( 
    bop_chunk_id: int,
    bop_im_id: int,
    obj_id: int,
    use_detections: bool,
    detections: Dict[Any, Any],
    max_num_preds: int,
    gt_object_annos: List[Any],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Collect info about object instances whose pose should be estimated."""

    # A list of per-instance dictionary with the input amodal bounding box,
    # input modal mask and GT annotation.
    instance_infos = []

    if use_detections:
        key = (bop_chunk_id, bop_im_id, obj_id)

        if key not in detections:
            logger.info(f"No detection found for {key}.")
            return []

        # Keep only the top N predictions.
        if len(detections[key]) == 1:
            # If there is only one prediction, there is no need to sort, also the next line won't work.
            preds = detections[key]
        else:
            preds = sorted(detections[key], key=lambda x: x["score"], reverse=True)[
                :max_num_preds
            ]

        # For every prediction, find the GT with the maximum IoU of masks.
        for pred in preds:
            # Amodal bounding box given by (x, y, w, h).
            box_amodal = np.array(pred["bbox"])

            mask_modal = rle_to_binary_mask(pred["segmentation"]).astype(
                np.uint8
            )
            mask_size = (mask_modal.shape[1], mask_modal.shape[0])

            # Apply the opening operation to remove isolated pixels.
            mask_modal = cv2.morphologyEx(
                mask_modal,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )

            # Calculate image shift (the input image may have been cropped to
            # become a multiple of ViT patch size).
            shift_x = 0
            if image_size[0] < mask_size[0]:
                shift_x = (mask_size[0] - image_size[0]) // 2
            elif image_size[0] > mask_size[0]:
                raise ValueError("Image is larger than mask.")
            shift_y = 0
            if image_size[1] < mask_size[1]:
                shift_y = (mask_size[1] - image_size[1]) // 2
            elif image_size[1] > mask_size[1]:
                raise ValueError("Image is larger than mask.")

            # Apply the shift to mask and box.
            mask_modal = mask_modal[shift_y:-shift_y, shift_x:-shift_x]
            box_amodal[0] -= shift_x
            box_amodal[1] -= shift_y

            # Convert box format from (x, y, w, h) to (x1, y1, x2, y2).
            box_amodal[2] += box_amodal[0]
            box_amodal[3] += box_amodal[1]

            best_anno_id = 0
            best_anno_iou = 0
            gt_anno = None
            if len(gt_object_annos) != 0:
                for anno_id, anno in enumerate(gt_object_annos):

                    # Calculate the IoU between the predicted mask and the GT mask.
                    iou = eval_errors.mask_iou(mask_modal, anno.masks_modal)

                    if iou > best_anno_iou:
                        best_anno_iou = iou
                        best_anno_id = anno_id

                gt_anno = gt_object_annos[best_anno_id]

            instance_infos.append(
                {
                    "input_box_amodal": box_amodal,
                    "input_mask_modal": mask_modal,
                    "gt_anno": gt_anno,
                    "gt_iou": best_anno_iou,
                    "time": pred["time"],
                }
            )

    else:
        # Use ground-truth.
        for anno in gt_object_annos:
            instance_infos.append(
                {
                    "input_box_amodal": anno.boxes_amodal.copy(),
                    "input_mask_modal": anno.masks_modal.copy(),
                    "gt_anno": anno,
                }
            )

    return instance_infos
