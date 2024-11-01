
# Read the json files that are the output of the infer.py script.
import os
import datetime
from collections import defaultdict

import numpy as np
import bop_toolkit_lib.config as bop_config
from bop_toolkit_lib import dataset_params

from utils import misc, logging, json_util

logger: logging.Logger = logging.get_logger()

# Load the estimated poses from the json file
object_dataset = "lmo"
version = "v1"
object_lids = None

signature = misc.slugify(object_dataset) + "_{}".format(version)
output_dir = os.path.join(
    bop_config.output_path, "inference", signature,
)

if object_lids is None:
    datasets_path = bop_config.datasets_path
    bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=object_dataset)
    object_lids = bop_model_props["obj_ids"]


detection_time_per_image = {}
run_time_per_image = defaultdict(float)
total_run_time = defaultdict(float)

for object_lid in object_lids:

    # Load the estimated poses from the json file
    results_path = os.path.join(output_dir, str(object_lid), "estimated-poses.json")
    logger.info("Loading estimated poses from: {}".format(results_path))

    estimated_poses = json_util.load_json(results_path)
    
    for estimated_pose_data in estimated_poses:
        scene_id = estimated_pose_data["scene_id"]
        img_id = estimated_pose_data["img_id"]
        obj_id = estimated_pose_data["obj_id"]
        inst_id = estimated_pose_data["inst_id"]
        cnos_time = estimated_pose_data["cnos_time"]

        detection_time_per_image[(scene_id, img_id)] = cnos_time

        run_time = 0
        for k, val in estimated_pose_data["time"].items():
            run_time += val
        
        run_time_per_image[(scene_id, img_id)] += run_time

    for scene_id, img_id in run_time_per_image.keys():
        total_run_time[(scene_id, img_id)] = run_time_per_image[(scene_id, img_id)] + detection_time_per_image[(scene_id, img_id)]

# BOP 19 format
lines = ["scene_id,im_id,obj_id,score,R,t,time"]
for object_lid in object_lids:

    # Load the estimated poses from the json file
    results_path = os.path.join(output_dir, str(object_lid), "estimated-poses.json")
    logger.info("Loading estimated poses from: {}".format(results_path))

    estimated_poses = json_util.load_json(results_path)

    for estimated_pose_data in estimated_poses:
        scene_id = estimated_pose_data["scene_id"]
        img_id = estimated_pose_data["img_id"]
        obj_id = estimated_pose_data["obj_id"]
        inst_id = estimated_pose_data["inst_id"]
        score = estimated_pose_data["score"]
        R = estimated_pose_data["R"]
        t = estimated_pose_data["t"]

        run_time = total_run_time[(scene_id, img_id)]

        lines.append(
            "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                scene_id=scene_id,
                im_id=img_id,
                obj_id=obj_id,
                score=score,
                R=" ".join(map(str, np.array(R).flatten().tolist())),
                t=" ".join(map(str, np.array(t).flatten().tolist())),
                time=run_time,
            )
        )

bop_path = os.path.join(output_dir, f"coarse_{object_dataset}-estimated-poses.csv")
logger.info("Saving BOP submission file to: {}".format(bop_path))
with open(bop_path, "wb") as f:
    f.write("\n".join(lines).encode("utf-8"))
