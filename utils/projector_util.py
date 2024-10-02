#!/usr/bin/env python3

from typing import Any, Dict, List, Optional

import numpy as np

import torch

from utils import logging
from utils.misc import array_to_tensor, tensor_to_array
from sklearn.decomposition import PCA


logger: logging.Logger = logging.get_logger()


class Projector:
    """An abstract class for a projector."""

    def fit(
        self,
        data_x: torch.Tensor,
        data_y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Fits the projector model to the training data.

        Args:
            data_x: A tensor of shape (num_features, feat_dims).
            data_y: A tensor of shape (num_features).
        """

        raise NotImplementedError

    def transform(self, data_x: torch.Tensor) -> torch.Tensor:
        """Transforms the provided data with the projector.

        Args:
            data_x: A tensor of shape (num_features, feat_dims).
        """
        raise NotImplementedError


class PCAProjector(Projector):
    def __init__(
        self, n_components: int, whiten: bool = False, **kwargs: Dict[str, Any]
    ) -> None:
        self.n_components: int = n_components
        self.whiten: bool = whiten
        self.pca: Any = PCA(n_components=n_components, whiten=whiten)

    def fit(
        self, data_x: torch.Tensor, data_y: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> None:

        # Cap the number of feature vectors for PCA fitting.
        if "max_samples" in kwargs:
            if data_x.shape[0] > kwargs["max_samples"]:
                perm = torch.randperm(data_x.shape[0])
                sampled_ids = perm[: kwargs["max_samples"]]
                data_x = data_x[sampled_ids]

        self.pca.fit(tensor_to_array(data_x))

    def transform(self, data_x: torch.Tensor) -> torch.Tensor:
        return array_to_tensor(self.pca.transform(tensor_to_array(data_x))).to(
            data_x.device
        )

def project_features(
    feat_vectors: torch.Tensor,
    projectors: List[Projector],
    batch_size: int = 4096,
) -> torch.Tensor:
    """Project feature vectors.

    Args:
        feat_vectors: A tensor of shape (num_features, feat_dims).
        projector: Feature projector.
    Return:
        Projected features.
    """

    for projector in projectors:
        feat_vectors = projector.transform(feat_vectors)

    return feat_vectors


def projector_to_tensordict(projector: Projector) -> Dict[str, Any]:
    """Converts a feature projector to a tensordict.

    Args:
        projector: Feature projector.
    Return:
        A tensordict.
    """

    if isinstance(projector, PCAProjector):
        return {
            "pca_projector": {
                "components": torch.tensor(projector.pca.components_),
                "explained_variance": torch.tensor(projector.pca.explained_variance_),
                "explained_variance_ratio": torch.tensor(projector.pca.explained_variance_ratio_),
                "singular_values": torch.tensor(projector.pca.singular_values_),
                "mean": torch.tensor(projector.pca.mean_),
                "noise_variance": torch.tensor(projector.pca.noise_variance_),
                "whiten": torch.tensor(projector.whiten),
            }
        }

    else:
        raise ValueError(f"Unknown projector type: {type(projector)}")

def projector_from_tensordict(projector_dict: Dict[str, Any]) -> Projector:
    """Converts a feature projector to a tensordict.

    Args:
        projector: Feature projector.
    Return:
        A tensordict.
    """

    if "pca_projector" in projector_dict:
        pca_projector = projector_dict["pca_projector"]

        pca = PCA(n_components=len(pca_projector["components"]))


        pca.components_ = tensor_to_array(pca_projector["components"])
        pca.explained_variance_ = tensor_to_array(pca_projector["explained_variance"])
        pca.explained_variance_ratio_ = tensor_to_array(pca_projector["explained_variance_ratio"])
        pca.singular_values_ = tensor_to_array(pca_projector["singular_values"])
        pca.mean_ = tensor_to_array(pca_projector["mean"])
        pca.noise_variance_ = tensor_to_array(pca_projector["noise_variance"])
        pca.whiten_ = tensor_to_array(pca_projector["whiten"])

        projector = PCAProjector(n_components=len(pca.components_), whiten=pca.whiten_)
        projector.pca = pca

        return projector

    else:
        raise ValueError("Unknown projector type.")
