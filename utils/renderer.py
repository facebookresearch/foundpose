from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import time

import numpy as np
import trimesh
import pyrender
from utils.misc import tensor_to_array
from utils import renderer_base, structs

from PIL import Image
import os.path as osp

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

class PyrenderRasterizer(renderer_base.RendererBase):
    """The base class which all renderers should inherit."""

    def __init__(
        self,
        renderer_flags: int = pyrender.constants.RenderFlags.NONE,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        self.renderer_flags = renderer_flags

        # Per-object scene (a special scene is created for each object model).
        self.object_scenes: Dict[int, pyrender.Scene] = {}
        
        self.object_meshes: Dict[int, trimesh.Trimesh] = {}

        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        self.im_size: Optional[Tuple[int, int]] = None

        if model_path:
            self.model_path = model_path

    def get_object_model(self,
        obj_id: int,
        mesh_color: Optional[structs.Color] = None,
        **kwargs: Any,
        ) -> trimesh.Trimesh:
        """Gets the object model.

        Args:
            obj_id: The object ID.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        # Load the object model.
        object_model_path = self.model_path.format(obj_id=obj_id)
        trimesh_model = trimesh.load(object_model_path)
        trimesh_model.vertices = trimesh_model.vertices/1000.0

        # Color the model.
        if mesh_color:
            num_vertices = trimesh_model.vertices.shape[0]
            trimesh_model.visual = trimesh.visual.objects.create_visual(
                vertex_colors=np.tile(mesh_color, (num_vertices, 1)),
                mesh=trimesh_model,
            )

        if obj_id not in self.object_meshes:
            self.object_meshes[obj_id]=trimesh_model

        return trimesh_model

    def add_object_model(
        self,
        obj_id: int,
        model_path: str,
        mesh_color: Optional[structs.Color] = None,
        **kwargs: Any,
    ) -> None:
        """Adds an object model to the renderer.

        Args:
            asset_key: The key of an asset to add to the renderer.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """
        
        if obj_id in self.object_scenes:
            return

        # Load the object model.
        if obj_id not in self.object_meshes:
            
            trimesh_model = trimesh.load(model_path)
            trimesh_model.vertices = trimesh_model.vertices/1000.0
            # Color the model.
            if mesh_color:
                num_vertices = trimesh_model.vertices.shape[0]
                trimesh_model.visual = trimesh.visual.objects.create_visual(
                    vertex_colors=np.tile(mesh_color, (num_vertices, 1)),
                    mesh=trimesh_model,
                )
            self.object_meshes[obj_id]=trimesh_model
        else:
            trimesh_model = self.object_meshes[obj_id]

        mesh = pyrender.Mesh.from_trimesh(trimesh_model)

        # Create a scene and add the model to the scene in the canonical pose.
        ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
        self.object_scenes[obj_id] = pyrender.Scene(
            bg_color=np.zeros(4), ambient_light=ambient_light
        )
        self.object_scenes[obj_id].add(mesh)

    def render_object_model(
        self,
        obj_id: int,
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:
        """Renders an object model in the specified pose.

        Args:
            camera_model_c2m: A camera model with the extrinsics set to a rigid
                transformation from the camera to the model frame.
            render_types: Types of images to render.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        # Create a scene for the object model if it does not exist yet.
        if obj_id not in self.object_scenes:
            self.add_object_model(obj_id)

        # Render the scene.
        return self._render_scene(
            scene_in_w=self.object_scenes[obj_id],
            camera_model_c2w=camera_model_c2w,
            render_types=render_types,
            return_tensors=return_tensors,
            debug=debug,
        )

    def render_meshes(
        self,
        meshes_in_w: Sequence[trimesh.Trimesh],
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[renderer_base.RenderType],
        mesh_colors: Optional[Sequence[structs.Color]] = None,
        return_tensors: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[renderer_base.RenderType, structs.ArrayData]:
        """Renders a list of meshes (see the base class)."""

        # Create a scene.
        ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
        scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)

        # Add meshes to the scene.
        for mesh_id, mesh in enumerate(meshes_in_w):
            # Scale the mesh from mm to m (expected by pyrender).
            mesh.vertices /= 1000.0,

            # Color the mesh.
            if mesh_colors:
                num_vertices = mesh.vertices.shape[0]
                mesh.visual = trimesh.visual.objects.create_visual(
                    vertex_colors=np.tile(mesh_colors[mesh_id], (num_vertices, 1)),
                    mesh=mesh,
                )

            # Add the mesh to the scene.
            scene.add(pyrender.Mesh.from_trimesh(mesh))

        # Render the scene.
        output = self._render_scene(
            scene_in_w=scene,
            camera_model_c2w=camera_model_c2w,
            render_types=render_types,
            return_tensors=return_tensors,
            debug=debug,
        )

        # Scale the meshes back to mm.
        for mesh in meshes_in_w:
            mesh.vertices *= 1000.0

        return output

    def _render_scene(
        self,
        scene_in_w: pyrender.Scene,
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[renderer_base.RenderType],
        return_tensors: bool = False,
        debug: bool = False,
    ) -> Dict[renderer_base.RenderType, structs.ArrayData]:
        """Renders an object model in the specified pose (see the base class)."""

        times = {}
        times["init_renderer"] = time.time()

        # Create the renderer if it does not exist yet, else check if the
        # rendering size is the one for which the renderer was created.
        if self.renderer is None:
            self.im_size = (camera_model_c2w.width, camera_model_c2w.height)
            self.renderer = pyrender.OffscreenRenderer(self.im_size[0], self.im_size[1])
        elif (
            self.im_size[0] != camera_model_c2w.width
            or self.im_size[1] != camera_model_c2w.height
        ):
            raise ValueError("All renderings must be of the same size.")

        times["init_renderer"] = time.time() - times["init_renderer"]
        times["init_scene"] = time.time()

        # OpenCV to OpenGL camera frame.
        trans_cv2gl = get_opencv_to_opengl_camera_trans()
        trans_c2w = camera_model_c2w.T_world_from_eye.dot(trans_cv2gl)

        # Convert translation from mm to m, as expected by pyrender.
        trans_c2w[:3, 3] *= 0.001

        # Camera for rendering.
        camera = pyrender.IntrinsicsCamera(
            fx=camera_model_c2w.f[0],
            fy=camera_model_c2w.f[1],
            cx=camera_model_c2w.c[0],
            cy=camera_model_c2w.c[1],
            znear=0.1,
            zfar=3000.0  
        )

        # Create a camera node.
        camera_node = pyrender.Node(camera=camera, matrix=trans_c2w)
        scene_in_w.add_node(camera_node)
        # Create light. 
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=2.4,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )

        light_node = pyrender.Node(light=light, matrix=trans_c2w)
        
        scene_in_w.add_node(light_node)

        times["init_scene"] = time.time() - times["init_scene"]
        times["render"] = time.time()

        # Rendering.
        color = None
        depth = None
        if self.renderer_flags & pyrender.RenderFlags.DEPTH_ONLY:
            assert self.renderer is not None
            depth = self.renderer.render(scene_in_w, flags=self.renderer_flags)
        else:
            assert self.renderer is not None
            color, depth = self.renderer.render(scene_in_w, flags=self.renderer_flags)

        times["render"] = time.time() - times["render"]
        times["postprocess"] = time.time()

        # Convert the RGB image from [0, 255] to [0.0, 1.0].
        if color is not None:
            color = color.astype(np.float32) / 255.0

        # Convert the depth map to millimeters.
        if depth is not None:
            depth *= 1000.0

        # Get the object mask.
        mask = None
        if renderer_base.RenderType.MASK in render_types:
            mask = depth > 0

        # Remove the camera so the scene contains only the object in the
        # canonical pose and can be reused.
        scene_in_w.remove_node(camera_node)
        scene_in_w.remove_node(light_node)

        # Prepare the output.
        output = {
            renderer_base.RenderType.COLOR: color,
            renderer_base.RenderType.DEPTH: depth,
            renderer_base.RenderType.MASK: mask,
        }
        if return_tensors:
            for name in output.keys():
                if output[name] is not None:
                    output[name] = misc.array_to_tensor(output[name])

        times["postprocess"] = time.time() - times["postprocess"]

        if debug:
            logger.info("PyrenderRasterizer run times:")
            for time_name, time_value in times.items():
                logger.info(f"- {time_name}: {time_value:.04f}s")

        return output


def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip


def get_opencv_to_opengl_camera_trans() -> np.ndarray:
    return get_opengl_to_opencv_camera_trans()
