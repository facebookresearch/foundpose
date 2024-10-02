#!/usr/bin/env python3

import os

from dataclasses import dataclass, field

from typing import Any, Dict, List, NamedTuple, Optional

import torch

from utils import projector_util, logging, misc
from utils.structs import PinholePlaneCameraModel

from utils.misc import tensor_to_array

logger: logging.Logger = logging.get_logger()


class FeatureOpts(NamedTuple):
    extractor_name: str


class TemplateDescOpts(NamedTuple):
    desc_type: str = "tfidf"

    # Options for tfidf template descriptor.
    tfidf_knn_metric: str = "l2"
    tfidf_knn_k: int = 3
    tfidf_soft_assign: bool = False
    tfidf_soft_sigma_squared: float = 10.0


@dataclass
class FeatureBasedObjectRepre:
    """Stores visual object features registered in 3D."""

    # 3D vertices of the object model of shape (num_vertices, 3).
    vertices: Optional[torch.Tensor] = None

    # 3D vertex normals of shape (num_vertices, 3).
    vertex_normals: Optional[torch.Tensor] = None

    # Feature vectors of shape (num_features, feat_dims).
    feat_vectors: Optional[torch.Tensor] = None

    # Feature options.
    feat_opts: Optional[FeatureOpts] = None

    # Mapping from feature to associated vertex ID of shape (num_features).
    feat_to_vertex_ids: Optional[torch.Tensor] = None

    # Mapping from feature to source template ID of shape (num_features).
    feat_to_template_ids: Optional[torch.Tensor] = None

    # Mapping from feature to assigned feature ID of shape (num_features).
    feat_to_cluster_ids: Optional[torch.Tensor] = None

    # Centroids of feature clusters of shape (num_clusters, feat_dims).
    feat_cluster_centroids: Optional[torch.Tensor] = None

    # Inverse document frequency (for tfidf template descriptors) of shape (num_clusters).
    # Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf
    feat_cluster_idfs: Optional[torch.Tensor] = None

    # Projectors of raw extracted features to features saved in `self.feat_vectors`.
    feat_raw_projectors: List[projector_util.Projector] = field(default_factory=list)

    # Projectors for visualizing features from `self.feat_vectors` (typically a PCA projector).
    feat_vis_projectors: List[projector_util.Projector] = field(default_factory=list)

    # Templates of shape (num_templates, channels, height, width).
    templates: Optional[torch.Tensor] = None

    # Per-template camera with extrinsics expressed as a transformation from the model space
    # to the camera space.
    template_cameras_cam_from_model: List[PinholePlaneCameraModel] = field(default_factory=list)

    # Template descriptors of shape (num_templates, desc_dims).
    template_descs: Optional[torch.Tensor] = None

    # Configuration of template descriptors.
    template_desc_opts: Optional[TemplateDescOpts] = None


def get_object_repre_dir_path(
    base_dir: str, repre_type: str, dataset: str, lid: int
) -> str:
    """Get a path to a directory where a representation of the specified object is stored."""

    return os.path.join(
        base_dir,
        dataset,
        repre_type,
        str(lid),
    )


def save_object_repre(
    repre: FeatureBasedObjectRepre,
    repre_dir: str,
) -> None:

    # Save the object into torch data.

    object_dict = {}

    for key, value in repre.__dict__.items():
        if value is not None and torch.is_tensor(value):
            object_dict[key] = value

    # Save camera metadata.
    object_dict["template_cameras_cam_from_model"] = []
    for camera in repre.template_cameras_cam_from_model:
        cam_data = {
            "f": torch.tensor(camera.f),
            "c": torch.tensor(camera.c),
            "width": camera.width,
            "height": camera.height,
            "T_world_from_eye": torch.tensor(camera.T_world_from_eye),
        }
        object_dict["template_cameras_cam_from_model"].append(cam_data)


    object_dict["feat_opts"] = repre.feat_opts._asdict()
    object_dict["template_desc_opts"] = repre.template_desc_opts._asdict()


    object_dict["feat_raw_projectors"] = []
    for projector in repre.feat_raw_projectors:
        object_dict["feat_raw_projectors"].append(projector_util.projector_to_tensordict(projector))

    object_dict["feat_vis_projectors"] = []
    for projector in repre.feat_vis_projectors:
        object_dict["feat_vis_projectors"].append(projector_util.projector_to_tensordict(projector))

    # Save the dictionary of tensors to the file
    repre_path = os.path.join(repre_dir, "repre.pth")
    logger.info(f"Saving repre to: {repre_path}")

    torch.save(object_dict, repre_path)

def load_object_repre(
    repre_dir: str,
    tensor_device: str = "cuda",
    load_fields: Optional[List[str]] = None,
) -> FeatureBasedObjectRepre:
    """Loads a representation of the specified object."""

    repre_path = os.path.join(repre_dir, "repre.pth")
    logger.info(f"Loading repre from: {repre_path}")
    object_dict = torch.load(repre_path)
    logger.info("Repre loaded.")

    repre_dict: Dict[str, Any] = {}

    for key, value in object_dict.items():
        if value is not None and (isinstance(value, torch.Tensor)):
            repre_dict[key] = value

    if object_dict["feat_opts"] is not None and (
        load_fields is None or "feat_opts" in load_fields
    ):
        repre_dict["feat_opts"] = FeatureOpts(**dict(object_dict["feat_opts"]))

    repre_dict["feat_raw_projectors"] = []
    if load_fields is None or "feat_raw_projectors" in load_fields:
        for projector in object_dict["feat_raw_projectors"]:
            repre_dict["feat_raw_projectors"].append(
                projector_util.projector_from_tensordict(projector)
            )

    repre_dict["feat_vis_projectors"] = []
    if load_fields is None or "feat_vis_projectors" in load_fields:
        for projector in object_dict["feat_vis_projectors"]:
            repre_dict["feat_vis_projectors"].append(
                projector_util.projector_from_tensordict(projector)
            )

    repre_dict["template_cameras_cam_from_model"] = []
    if load_fields is None or "template_cameras_cam_from_model" in load_fields:
        for camera in object_dict["template_cameras_cam_from_model"]:
            repre_dict["template_cameras_cam_from_model"].append(
                    PinholePlaneCameraModel(
                        f=camera["f"], ## needs conversion
                        c=camera["c"], ## needs conversion
                        width=camera["width"],
                        height=camera["height"],
                        T_world_from_eye=camera["T_world_from_eye"], ## needs conversion
                    )
            )

    if load_fields is None or "template_desc_opts" in load_fields:
        if object_dict["template_desc_opts"] is not None:
            repre_dict["template_desc_opts"] = TemplateDescOpts(
                **dict(object_dict["template_desc_opts"])
            )

    # Convert to the corresponding Python structure.
    repre = FeatureBasedObjectRepre(**repre_dict)

    # Optionally move tensors to GPU.
    if tensor_device != "cpu":

        def move_to_device(x: torch.Tensor) -> torch.Tensor:
            return x.to(tensor_device)

        repre = misc.map_fields(move_to_device, repre, only_type=torch.Tensor)

    return repre


def convert_object_repre_to_numpy(
    repre: FeatureBasedObjectRepre,
) -> FeatureBasedObjectRepre:

    repre_out = FeatureBasedObjectRepre()
    for name, value in repre.__dict__.items():
        if value is not None and isinstance(value, torch.Tensor):
            value = tensor_to_array(value)
        setattr(repre_out, name, value)

    return repre_out
