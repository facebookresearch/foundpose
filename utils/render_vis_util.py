
#!/usr/bin/env python3

"""Visualization functions."""

import colorsys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

from utils import logging, structs, geometry, misc

from utils import renderer_base, json_util

logger: logging.Logger = logging.get_logger()

# Colors of the left (0) and right (1) hands.
DEFAULT_FG_OPACITY = 0.5
DEFAULT_BG_OPACITY = 0.7
DEFAULT_FONT_SIZE = 15
DEFAULT_TXT_OFFSET = (2, 1)

def write_text_on_image(
    im: np.ndarray,
    txt_list: Sequence[Dict[str, Any]],
    loc: Tuple[int, int] = DEFAULT_TXT_OFFSET,
    color: structs.Color = (1.0, 1.0, 1.0),
    size: int = 20,
) -> np.ndarray:
    """Writes text on an image.

    Args:
        im: An image on which to write the text.
        txt_list: A list of dictionaries, each describing one text line:
        - "name": A text info.
        - "val": A value.
        - "fmt": A string format for the value.
        loc: A location of the top left corner of the text box.
        color: A font color.
        size: A font size.
    Returns:
        The input image with the text written on it.
    """

    im_pil = Image.fromarray(im)

    # Load the font.
    font = ImageFont.load("arial.pil")

    # Clip the text location to the image.
    im_size = (im.shape[1], im.shape[0])
    loc = tuple(
        misc.clip_2d_point(torch.as_tensor(loc), torch.as_tensor(im_size))
    )

    # Write the text.
    draw = ImageDraw.Draw(im_pil)
    for info in txt_list:
        txt = ""
        if "name" in info:
            txt += info["name"]
        if "val" in info:
            # Determine the print format.
            if "fmt" in info:
                val_tpl = "{" + info["fmt"] + "}"
            elif type(info["val"]) == float:
                val_tpl = "{:.3f}"
            else:
                val_tpl = "{}"
            if txt != "":
                txt += ": "
            txt += val_tpl.format(info["val"])
        draw.text(
            xy=loc,
            text=txt,
            fill=tuple([int(255 * c) for c in color]),
            font=font,
        )
        _, text_height = font.getsize("X")
        loc = (loc[0], loc[1] + int(1.3 * text_height))
    del draw

    return np.array(im_pil)


def vis_meshes(
    base_image: np.ndarray,
    meshes_in_w: List[trimesh.Trimesh],
    colors: Optional[Sequence[structs.Color]] = None,
    stickers: Optional[Sequence[str]] = None,
    camera_c2w: Optional[structs.CameraModel] = None,
    renderer: Optional[renderer_base.RendererBase] = None,
    fg_opacity: float = DEFAULT_FG_OPACITY,
    bg_opacity: float = DEFAULT_BG_OPACITY,
    font_size: int = DEFAULT_FONT_SIZE,
    all_in_one: bool = True,
) -> List[np.ndarray]:
    """Renders 3D meshes on top of an image.

    Args:
        base_image: Image to draw the meshes on.
        meshes_in_w: List of meshes in the world space.
        colors: List of mesh colors (each mesh is colored with a single color).
        stickers: List of mesh stickers to write on the image.
        camera_c2w: Camera to render the meshes to.
        renderer: For rendering the meshes.
        fg_opacity: Opacity of the foreground (i.e. meshes).
        bg_opacity: Opacity of the bacgkround (i.e. the base image).
        font_size: Font size of the text to show.
        all_in_one: Whether to create a single visualization image with all the
            meshes or one visualization image per mesh.
    Returns:
        List of visualizations.
    """

    if colors is not None and len(meshes_in_w) != len(colors):
        raise ValueError("Number of meshes and colors must be the same.")
    if stickers is not None and len(meshes_in_w) != len(stickers):
        raise ValueError("Number of meshes and stickers must be the same.")
    if renderer is None or camera_c2w is None:
        raise ValueError("Renderer or camera model not provided.")

    # Make sure the base image has three channels.
    base_image_3ch = misc.ensure_three_channels(base_image)

    vis_tiles = []
    meshes_in_w_per_tile = [meshes_in_w] if all_in_one else [[m] for m in meshes_in_w]
    for tile_meshes_in_w in meshes_in_w_per_tile:
        renderings = renderer.render_meshes(
            meshes_in_w=tile_meshes_in_w,
            camera_model_c2w=camera_c2w,
            mesh_colors=colors,
            render_types=[
                renderer_base.RenderType.COLOR,
                renderer_base.RenderType.MASK,
            ],
        )
        # Alpha channel of the rendered meshes.
        fg_mask = np.expand_dims(
            renderings[renderer_base.RenderType.MASK] > 0, axis=-1
        ).astype(np.float32)

        # Scale color at each pixel such as there is enough space for the
        # rendered color (so the color values do not overflow after blending).
        masked_fg_opacity = fg_mask * fg_opacity
        base_image_weights = np.minimum(
            bg_opacity * np.ones_like(fg_mask, dtype=np.float32),
            1.0 - masked_fg_opacity,
        )

        # Blend the base image with the rendered meshes.
        ren_color: np.ndarray = np.asarray(renderings[renderer_base.RenderType.COLOR])
        vis_tile = (
            base_image_3ch.astype(np.float32) * base_image_weights
            + 255 * ren_color * masked_fg_opacity
        ).astype(np.uint8)

        # Show stickers (with info about the meshes) if provided.
        if stickers is not None:
            for mesh_id in range(len(tile_meshes_in_w)):

                # Place the sticker at the projection of the object centroid.
                sticker_loc = camera_c2w.world_to_window(
                    tile_meshes_in_w[mesh_id].centroid
                )

                vis_tile = write_text_on_image(
                    vis_tile,
                    [{"name": stickers[mesh_id]}],
                    loc=sticker_loc,
                    size=font_size,
                )

        vis_tiles.append(vis_tile)

    return vis_tiles

def vis_posed_meshes_of_objects(
    base_image: np.ndarray,
    object_lids: Sequence[int],
    object_poses_m2w: Sequence[structs.ObjectPose],
    camera_c2w: structs.CameraModel,
    renderer: renderer_base.RendererBase,
    object_colors: Optional[Sequence[structs.Color]] = None,
    object_stickers: Optional[Sequence[str]] = None,
    fg_opacity: float = DEFAULT_FG_OPACITY,
    bg_opacity: float = DEFAULT_BG_OPACITY,
    font_size: int = DEFAULT_FONT_SIZE,
    all_in_one: bool = True,
) -> List[np.ndarray]:
    """Visualizes meshes of objects in the specified poses.

    An object is specified by corresponding items of `object_lids` and
    `object_poses_m2w`.

    Args:
        base_image: Image to draw the meshes on.
        object_lids: List of asset keys of objects to visualize.
        object_poses_m2w: List of object poses (from model to world).
        camera_c2w: Camera to render the meshes to.
        renderer: For rendering the meshes.
        object_colors: List of object colors (each mesh has a single color).
        object_stickers: List of object stickers.
        fg_opacity: Opacity of the foreground (i.e. meshes).
        bg_opacity: Opacity of the bacgkround (i.e. the base image).
        font_size: Font size of the text to show.
        all_in_one: Whether to create a single visualization image with all the
            meshes or one visualization image per mesh.
    Returns:
        List of visualizations.
    """

    if len(object_lids) != len(object_poses_m2w):
        raise ValueError("Input lists `object_*` must have the same length.")

    # Collect object meshes in the world frame.
    object_meshes_in_w = []
    for object_idx, obj_lid in enumerate(object_lids):

        object_mesh = renderer.get_object_model(obj_lid).copy()

        obj_vertices = object_mesh.vertices*1000

        object_mesh.vertices = geometry.transform_3d_points_numpy(
            misc.get_rigid_matrix(object_poses_m2w[object_idx]), obj_vertices
        )

        # apply the camera pose as well, then give renderer I. 
        object_meshes_in_w.append(object_mesh)

     # Collect stickers to be shown on the image at the meshes.
    stickers = None
    if object_stickers is not None:
        stickers = list(object_stickers)

    camera_c2w_input = camera_c2w.copy()
    camera_c2w_input.T_world_from_eye = np.linalg.inv(camera_c2w.T_world_from_eye)

    return vis_meshes(
        base_image=base_image,
        meshes_in_w=object_meshes_in_w,
        colors=list(object_colors),
        stickers=stickers,
        camera_c2w=camera_c2w_input,
        renderer=renderer,
        fg_opacity=fg_opacity,
        bg_opacity=bg_opacity,
        all_in_one=all_in_one,
    )


def create_object_mask(
    base_image: np.ndarray,
    object_lids: Sequence[int],
    object_poses_m2w: Sequence[structs.ObjectPose],
    camera_c2w: structs.CameraModel,
    renderer: renderer_base.RendererBase,
    object_colors: Optional[Sequence[structs.Color]] = None,
    object_stickers: Optional[Sequence[str]] = None,
    fg_opacity: float = DEFAULT_FG_OPACITY,
    bg_opacity: float = DEFAULT_BG_OPACITY,
    font_size: int = DEFAULT_FONT_SIZE,
    all_in_one: bool = True,
) -> List[np.ndarray]:
    """Visualizes meshes of objects in the specified poses.

    An object is specified by corresponding items of `object_lids` and
    `object_poses_m2w`.

    Args:
        base_image: Image to draw the meshes on.
        object_lids: List of asset keys of objects to visualize.
        object_poses_m2w: List of object poses (from model to world).
        camera_c2w: Camera to render the meshes to.
        renderer: For rendering the meshes.
        object_colors: List of object colors (each mesh has a single color).
        object_stickers: List of object stickers.
        fg_opacity: Opacity of the foreground (i.e. meshes).
        bg_opacity: Opacity of the bacgkround (i.e. the base image).
        font_size: Font size of the text to show.
        all_in_one: Whether to create a single visualization image with all the
            meshes or one visualization image per mesh.
    Returns:
        List of visualizations.
    """

    if len(object_lids) != len(object_poses_m2w):
        raise ValueError("Input lists `object_*` must have the same length.")

    obj_lid = object_lids[0]
    object_idx = 0
    image_size = base_image.shape[:2]

    object_mesh = renderer.get_object_model(obj_lid).copy()

    obj_vertices = object_mesh.vertices*1000

    vertices = geometry.transform_3d_points_numpy(
        misc.get_rigid_matrix(object_poses_m2w[object_idx]), obj_vertices
    )
    
    projected_points = camera_c2w.world_to_window(vertices)
   
    image = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)

    # Round the projected points to nearest integer pixel indices
    projected_points_int = np.round(projected_points).astype(int)

    h, w = image_size
    for pt in projected_points_int:
        x, y = pt
        # Check if the point is within image bounds
        if 0 <= x < w and 0 <= y < h:
            image[y, x] = [0, 0, 0]  # Set the pixel to black (BGR format)
    
    return image

    # return vis_meshes(
    #     base_image=base_image,
    #     meshes_in_w=object_meshes_in_w,
    #     colors=list(object_colors),
    #     stickers=stickers,
    #     camera_c2w=camera_c2w,
    #     renderer=renderer,
    #     fg_opacity=fg_opacity,
    #     bg_opacity=bg_opacity,
    #     all_in_one=all_in_one,
    # )