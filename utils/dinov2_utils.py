#!/usr/bin/env python3

import math
import types
import typing as tp

from typing import Callable, List, Optional, Tuple

import dinov2.hub.backbones as dinov2_backbones

import torch
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T

from utils import logging

from torch import nn
from torch.utils.hooks import RemovableHandle
from torchinfo import summary

logger: logging.Logger = logging.get_logger()


class DinoFeatureExtractor(nn.Module):
    """DINOv2 feature extractor.

    Some methods are copied/adapted from (distributed under MIT License):
    https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py
    """

    def __init__(self, model_name: str) -> None:
        """DINOv2 constructor.

        Args:
            model_name: Name of DINOv2 variant, potentially including also values
                of selected parameters. Two name formats are currently supported:
                (1) "dinov2_<version>" (e.g., "dinov2_vits14-reg")
                (2) "dinov2_version=<version>_stride=<stride>_facet=<facet>...
                    ..._layer=<layer>_norm=<norm> where:
                    - version: One of {"vits14", "vitb14", "vitl14", "vitg14",
                      "vits14-reg", "vitb14-reg", "vitl14-reg", "vitg14-reg"}.
                    - stride: Stride of the backbone (DINOv2 was trained with stride
                      14px, but the stride can be changed at inference).
                    - facet: One of {"token", "key", "value", "query", "attn"}.
                    - norm: One of {0, 1}. Whether to apply LayerNorm to the output.

        """

        super().__init__()

        # Default parameter values.
        self.version: str = "vits14-reg"
        self.stride: int = 14
        self.facet: str = "token"
        self.layer: int = 9
        self.apply_norm: bool = True

        # Parse the model name.
        name_items = model_name.split("_")
        assert name_items[0] == "dinov2"
        if len(name_items) == 2:
            # Example: "dinov2_vits14"
            self.version = name_items[1]
        else:
            # Example: "dinov2_version=vitl14_stride=14_facet=key_layer=18_norm=1"
            for item in name_items[1:]:
                name, value = item.split("=")
                if name == "version":
                    self.version = value
                elif name == "stride":
                    self.stride = int(value)
                elif name == "facet":
                    self.facet = value
                elif name == "layer":
                    self.layer = int(value)
                elif name == "norm":
                    self.apply_norm = bool(int(value))

        # Build the base model.
        self.model_base_name: str = f"dinov2_{self.version}".replace("-", "_")
        self.model: torch.nn.Module = dinov2_backbones.__dict__[self.model_base_name](
            pretrained=True
        )

        # Load pre-trained weights.
        # path = _DINOV2_BASE_URL + f"{self.model_base_name}_pretrain.pth"
        # logger.info(f"Loading DINOv2 weights from: {path}")
        # with g_pathmgr.open(path, mode="rb") as f:
        #     state_dict = torch.load(f, map_location="cpu")
        # self.model.load_state_dict(state_dict, strict=True)

        # DINOv2 is trained with stride 14.
        if self.stride != 14:
            self.model = self.patch_vit_resolution(self.model, stride=self.stride)

        self.patch_size: int = self.model.patch_embed.patch_size[0]
        self.stride = self.model.patch_embed.proj.stride[0]

        self.model.eval()

        self._feats: List[torch.Tensor] = []
        self.hook_handlers: List[RemovableHandle] = []
        self.num_patches: Optional[Tuple[int, int]] = None

        # logger.info(f"Summary of the {model_name} feature extractor")
        # summary(self.model)

        # Image preprocessor.
        # Normalize with the ImageNet mean/std.
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def forward(self, images: torch.Tensor) -> tp.Dict[str, torch.Tensor]:

        # Note: function `extract_output_features` defined in the DINOv2 model itself
        # outputs normalized token facets from the last layer. For example, for
        # vitl14, the same output can be obtained with model name:
        # "dinov2_version=vitl14_stride=14_facet=token_layer=23_norm=1"

        # Normalize the input image.
        images = self.normalize(images)

        outputs = self.extract_descriptors(
            batch=images,
            layer=self.layer,
            facet=self.facet,
        )

        # CLS tokens of size Bx1xD.
        cls_tokens = outputs["cls_tokens"][:, 0, :, :]

        # Patch tokens of size BxTxD.
        patch_tokens = outputs["patch_tokens"][:, 0, :, :]

        # Normalize the tokens (LayerNorm is applied to all of them, as in DINOv2).
        if self.apply_norm:
            tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
            tokens = self.model.norm(tokens)
            cls_tokens = tokens[:, :1, :]
            patch_tokens = tokens[:, 1:, :]

        # Reshape patch tokens to BxDxHxW.
        bsz, _, w, h = images.shape
        d = patch_tokens.shape[-1]
        num_patches = (
            1 + (h - self.patch_size) // self.stride,
            1 + (w - self.patch_size) // self.stride,
        )
        feature_maps = patch_tokens.reshape(
            bsz, num_patches[1], num_patches[0], d
        ).permute(0, 3, 1, 2)

        return {
            "cls_tokens": cls_tokens[:, 0, :],  # BxD
            "feature_maps": feature_maps,  # BxDxHxW
        }

    def _get_hook(
        self, facet: str
    ) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(
                module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
            ) -> None:
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx: int = 0
        elif facet == "key":
            facet_idx: int = 1
        elif facet == "value":
            facet_idx: int = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """Registers hook to extract features.

        Args:
            layers: Layers from which to extract features.
            facet: Facet to extract. One of the following options:
                {"key", "query", "value", "token", "attn"}
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """Unregisters the hooks. should be called after feature extraction."""

        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self,
        batch: torch.Tensor,
        layers: Optional[List[int]] = None,
        facet: str = "key",
    ) -> List[torch.Tensor]:
        """Extracts features.

        Args:
            batch: Batch to extract features for. Has shape BxCxHxW.
            layers: Layer to extract.
            facet: Facet to extract.
        Returns:
            Tensor of features.
                If facet is "key" | "query" | "value" has shape Bxhxtxd.
                If facet is "attn" has shape Bxhxtxt.
                If facet is "token" has shape Bxtxd.
        """

        if layers is None:
            layers = [11]

        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.patch_size) // self.stride,
            1 + (W - self.patch_size) // self.stride,
        )
        return self._feats

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: int = 11,
        facet: str = "key",
    ) -> tp.Dict[str, torch.Tensor]:
        """Extracts descriptors from the model.

        Args:
            batch: Batch to extract descriptors for. Has shape BxCxHxW.
            layers: Layer to extract.
            facet: Facet to extract.
        Returns:
            Tensor of descriptors. Bx1xtxd' where d is the descriptor dimension.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors."""

        self._extract_features(batch, [layer], facet)
        outputs = {}

        x = self._feats[0]

        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd

        outputs["cls_tokens"] = (
            x[:, :, [0], :]
            .permute(0, 2, 3, 1)
            .flatten(start_dim=-2, end_dim=-1)
            .unsqueeze(dim=1)
        )

        # Remove the CLS token and register tokens.
        x = x[:, :, (self.model.num_register_tokens + 1) :, :]

        # Bx1xtx(dxh)
        outputs["patch_tokens"] = (
            x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
        )

        return outputs

    def _fix_pos_enc(
        self, patch_size: int, stride_hw: Tuple[int, int]
    ) -> Callable[[torch.Tensor, int, int], torch.Tensor]:
        """Creates a method for position encoding interpolation.

        Args:
            patch_size: Patch size of the model.
            stride_hw: A tuple containing new height and width stride respectively.
        Returns:
            The interpolation method.
        """

        def interpolate_pos_encoding(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""

            # A small number is added to avoid floating point error in the interpolation
            # (see discussion at https://github.com/facebookresearch/dino/issues/8).
            w0, h0 = w0 + 0.1, h0 + 0.1

            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    def patch_vit_resolution(self, model: nn.Module, stride: int) -> nn.Module:
        """Change resolution of model output by changing the stride of the patch extraction.

        Args:
            model: The model to change resolution for.
            stride: The new stride parameter.
        Returns:
            The adjusted model.
        """
        patch_size = model.patch_embed.patch_size[0]
        assert model.patch_embed.patch_size == model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            (patch_size // s_) * s_ == patch_size for s_ in stride
        ), f"stride {stride} should divide patch_size {patch_size}"

        # Fix the stride.
        model.patch_embed.proj.stride = stride

        # Fix the positional encoding code.
        model.interpolate_pos_encoding = types.MethodType(
            self._fix_pos_enc(patch_size, stride), model
        )
        return model
