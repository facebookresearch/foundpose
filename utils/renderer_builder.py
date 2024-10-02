#!/usr/bin/env python3

"""Utilities for building renderers."""

from enum import Enum
from typing import Any, Optional

from utils import renderer_base


class RendererType(Enum):
    """The renderer to be used."""

    PYRENDER_RASTERIZER = "pyrender_rasterizer"


def build(
    renderer_type: RendererType,
    **kwargs: Any,
) -> renderer_base.RendererBase:
    """Builds renderer given the render name.

    Args:
        renderer_type: Name of the renderer.
        asset_library: Optional asset library to initialize the renderer with.
    Returns:
        A model.
    """
    if renderer_type == RendererType.PYRENDER_RASTERIZER:
        from utils import renderer

        return renderer.PyrenderRasterizer(**kwargs)
    else:
        raise ValueError(f"Unknown renderer `{renderer_type}`.")
