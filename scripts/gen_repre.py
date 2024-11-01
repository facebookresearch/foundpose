#!/usr/bin/env python3

"""Generates a feature-based object representation."""

import os
import logging

from typing import Any, Dict, List, NamedTuple, Optional

import torch

from utils.misc import array_to_tensor

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from utils import (
    cluster_util,
    feature_util,
    projector_util,
    repre_util,
    template_util,
    preprocess_util,
    config_util,
    json_util,
    logging,
    misc,
    structs,
)

from utils.structs import PinholePlaneCameraModel

import numpy as np

class GenRepreOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    templates_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None

    # Feature extraction options.
    extractor_name: str = "dinov2_vits14_reg"
    grid_cell_size: float = 14.0

    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # Feature clustering options.
    cluster_features: bool = True
    cluster_num: int = 2048

    # Template descriptor options.
    template_desc_opts: Optional[repre_util.TemplateDescOpts] = None

    # Other options.
    overwrite: bool = True
    debug: bool = True


def generate_raw_repre(
    opts: GenRepreOpts,
    object_dataset: str,
    object_lid: int,
    extractor: torch.nn.Module,
    output_dir: str,
    device: str = "cuda",
    debug: bool = False,
) -> repre_util.FeatureBasedObjectRepre:

    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Prepare a timer.
    timer = misc.Timer(enabled=debug)

    datasets_path = bop_config.datasets_path

    # Load the template metadata.
    # metadata_path = "/Users/evinpinar/Documents/opensource_foundpose/output/templates/v1/lmo/1/metadata.json"
    metadata_path = os.path.join(
        bop_config.output_path,
        "templates",
        opts.templates_version,
        opts.object_dataset,
        str(object_lid),
        "metadata.json"
    )
    metadata = json_util.load_json(metadata_path)

    # Prepare structures for storing data.
    feat_vectors_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_cameras_cam_from_model_list = []

    # Use template images specified in the metadata.
    template_id = 0
    num_templates = len(metadata)
    for data_id, data_sample in enumerate(metadata):

        logger.info(
            f"Processing dataset {data_id}/{num_templates}, "
        )

        timer.start()

        camera_sample = data_sample["cameras"]
        camera_world_from_cam = PinholePlaneCameraModel(
                width=camera_sample["ImageSizeX"],
                height=camera_sample["ImageSizeY"],
                f=(camera_sample["fx"],camera_sample["fy"]),
                c=(camera_sample["cx"],camera_sample["cy"]),
                T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"])
            )

        # RGB/monochrome and depth images (in mm).
        image_path = data_sample["rgb_image_path"]
        depth_path = data_sample["depth_map_path"]
        mask_path = data_sample["binary_mask_path"]

        image_arr = inout.load_im(image_path) # H,W,C
        depth_image_arr = inout.load_depth(depth_path)
        mask_image_arr = inout.load_im(mask_path)

        image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2,0,1).to(device) / 255.0
        depth_image_hw = array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(mask_image_arr).to(torch.float32).to(device)

        # Get the object annotation.
        assert data_sample["dataset"] == object_dataset
        assert data_sample["lid"] == object_lid
        assert data_sample["template_id"] == data_id

        object_pose = data_sample["pose"]

        # Transformations.
        object_pose_rigid_matrix = np.eye(4)
        object_pose_rigid_matrix[:3, :3] = object_pose["R"]
        object_pose_rigid_matrix[:3, 3:] = object_pose["t"]
        T_world_from_model = (
            array_to_tensor(object_pose_rigid_matrix)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = (
            array_to_tensor(camera_world_from_cam.T_world_from_eye)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        timer.elapsed("Time for getting template data")
        timer.start()

        # Extract features from the current template.
        (
            feat_vectors,
            feat_to_vertex_ids,
            vertices_in_model,
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera_world_from_cam,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )

        timer.elapsed("Time for feature extraction")
        timer.start()

        # Store data.
        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids = template_id * torch.ones(
            feat_vectors.shape[0], dtype=torch.int32, device=device
        )
        feat_to_template_ids_list.append(feat_to_template_ids)

        # Save the template as uint8 to save space.
        image_chw_uint8 = (image_chw * 255).to(torch.uint8)
        templates_list.append(image_chw_uint8)

        # Store camera model of the current template.
        camera_model = camera_world_from_cam.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

        # Increment the template ID.
        template_id += 1

        timer.elapsed("Time for storing data")

    logger.info("Processing done.")

    # Build the object representation from the collected data.
    return repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_cam_from_model_list,
    )


def generate_repre(
    opts: GenRepreOpts,
    dataset: str,
    lid: int,
    device: str = "cuda",
    extractor: Optional[torch.nn.Module] = None,
) -> None:

    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    datasets_path = bop_config.datasets_path

    # Prepare a timer.
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Prepare the output folder.
    base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
    output_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a JSON file.
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # Prepare a feature extractor.
    if extractor is None:
        extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)

    timer.elapsed("Time for preparation")
    timer.start()

    # Build raw object representation.
    repre = generate_raw_repre(
        opts=opts,
        object_dataset=dataset,
        object_lid=lid,
        extractor=extractor,
        output_dir=output_dir,
        device=device,
    )

    feat_vectors = repre.feat_vectors
    assert feat_vectors is not None

    timer.elapsed("Time for generating raw representation")

    # Optionally transform the feature vectors to a PCA space.
    if opts.apply_pca:
        timer.start()

        # Prepare a PCA projector.
        logger.info("Preparing PCA...")
        pca_projector = projector_util.PCAProjector(
            n_components=opts.pca_components, whiten=opts.pca_whiten
        )
        pca_projector.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_raw_projectors.append(pca_projector)

        # Transform the selected feature vectors to the PCA space.
        feat_vectors = pca_projector.transform(feat_vectors)

        timer.elapsed("Time for PCA")

    # Cluster features into visual words.
    if opts.cluster_features:
        timer.start()

        logger.info(f"Clustering features into {opts.cluster_num} visual words...")
        centroids, cluster_ids, centroid_distances = cluster_util.kmeans(
            samples=feat_vectors,
            num_centroids=opts.cluster_num,
            verbose=True,
        )

        # Store the clustering results in the object repre.
        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids

        # Get cluster sizes.
        unique_ids, unique_counts = torch.unique(cluster_ids, return_counts=True)

        timer.elapsed("Time for feature clustering")
        logging.log_heading(
            logger,
            f"{feat_vectors.shape[0]} feature vectors were clustered into {len(centroids)} clusters "
            f"with {unique_counts.min()} to {unique_counts.max()} elements.",
        )

    # Generate template descriptors.
    if opts.template_desc_opts is not None:
        timer.start()

        repre.template_desc_opts = opts.template_desc_opts

        # Calculate tf-idf descriptors.
        if opts.template_desc_opts.desc_type == "tfidf":

            assert feat_vectors is not None
            assert repre.feat_cluster_centroids is not None
            assert repre.feat_to_cluster_ids is not None
            assert repre.feat_to_template_ids is not None
            assert repre.templates is not None

            repre.template_descs, repre.feat_cluster_idfs = (
                template_util.calc_tfidf_descriptors(
                    feat_vectors=feat_vectors,
                    feat_words=repre.feat_cluster_centroids,
                    feat_to_word_ids=repre.feat_to_cluster_ids,
                    feat_to_template_ids=repre.feat_to_template_ids,
                    num_templates=len(repre.templates),
                    tfidf_knn_k=opts.template_desc_opts.tfidf_knn_k,
                    tfidf_soft_assign=opts.template_desc_opts.tfidf_soft_assign,
                    tfidf_soft_sigma_squared=opts.template_desc_opts.tfidf_soft_sigma_squared,
                )
            )

        else:
            raise ValueError(
                f"Unknown template descriptor type: {opts.template_desc_opts.desc_type}"
            )

        timer.elapsed("Time for generating template descriptors")

    timer.start()

    # Create a PCA projector for visualization purposes (or reuse an existing one).
    if len(repre.feat_raw_projectors) and isinstance(
        repre.feat_raw_projectors[0], projector_util.PCAProjector
    ):
        repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]
    else:
        # Prepare a PCA projector.
        num_pca_dims_vis = 3
        pca_projector_vis = projector_util.PCAProjector(
            n_components=num_pca_dims_vis, whiten=False
        )
        pca_projector_vis.fit(
            feat_vectors, max_samples=opts.pca_max_samples_for_fitting
        )
        repre.feat_vis_projectors = [pca_projector_vis]

    repre.feat_vectors = feat_vectors

    timer.elapsed("Time for finding PCA for visualizations")
    timer.start()

    # Save the generated object representation.
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    repre_util.save_object_repre(repre, repre_dir)

    timer.elapsed("Time for saving the object representation")


def generate_repre_from_list(opts: GenRepreOpts) -> None:

    # Get IDs of objects to process.
    object_lids = opts.object_lids
    if object_lids is None:
        datasets_path = bop_config.datasets_path
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
        object_lids = bop_model_props["obj_ids"]

    # Prepare a feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    # Process each image separately.
    for object_lid in object_lids:
        generate_repre(opts, opts.object_dataset, object_lid, device, extractor)


def main() -> None:
    generate_repre_from_list(
        config_util.load_opts_from_json_or_command_line(GenRepreOpts)[0]
    )


if __name__ == "__main__":
    main()
