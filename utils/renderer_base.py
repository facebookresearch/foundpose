#!/usr/bin/env python3


"""The base class for renderers."""


from enum import Enum
from typing import Any, Dict, Optional, Sequence

import numpy as np
import trimesh
from utils import structs 


def get_single_model_color(mesh: trimesh.Trimesh) -> structs.Color:
    """Gets a single color for a mesh.

    Args:
        mesh: A mesh for which to get the color.
    Returns:
        A single color for the mesh -- either the average vertex color (if vertex
        colors are defined) or a default color.
    """

    try:
        return tuple(np.mean(mesh.visual.vertex_colors[:, :3], axis=0) / 255.0)
    except AttributeError:
        return (0.0, 1.0, 0.0)


class RenderType(Enum):
    """The rendering type.

    COLOR: An RGB image.
    DEPTH: A depth image with the depth values in mm.
    NORMAL: A normal map with normals expressed in the camera space.
    MASK: A binary mask.
    """

    COLOR = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    MASK = "mask"


class RendererBase:
    """The base class which all renderers should inherit."""

    def __init__(self, **kwargs: Any) -> None:
        
        raise NotImplementedError()

    def add_object_model(
        self,
        object_id: int,
        mesh_color: Optional[structs.Color] = None,
        **kwargs: Any,
    ) -> None:
        """Adds an object model to the renderer.

        Args:
            asset_key: The key of an asset to add to the renderer.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        raise NotImplementedError()

    def render_object_model(
        self,
        object_id: int,
        camera_model_c2m: structs.CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:
        """Renders an object model in the specified pose.

        Args:
            asset_key: The key of an asset to render.
            camera_model_c2m: A camera model with the extrinsics set to a rigid
                transformation from the camera to the model frame.
            render_types: Types of images to render.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        raise NotImplementedError()

    def render_meshes(
        self,
        meshes_in_w: Sequence[trimesh.Trimesh],
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[RenderType],
        mesh_colors: Optional[Sequence[structs.Color]] = None,
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:
        """Renders a list of meshes.

        Args:
            meshes_in_w: A list of meshes to render. The meshes are assumed to be
                defined in the same world coordinate frame.
            camera_model_c2w: A camera model with the extrinsics set to a rigid
                transformation from the camera to the world frame.
            render_types: Types of images to render.
            mesh_colors: A single color per mesh. Original mesh colors are used
                if not specified.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        raise NotImplementedError()
