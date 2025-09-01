# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from typing import Any, Optional, Tuple

import faiss
import faiss.contrib.torch_utils
import torch

class KNN:
    """K nearest neighbor search.

    References:
    [1] towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
    """

    def __init__(
        self,
        k: int = 1,
        metric: str = "l2",
        radius: Optional[float] = None,
        res: Optional[Any] = None,
    ) -> None:
        """
        Args:
            k: The number of nearest neighbors to return.
            metric: The distance metric to use. Can be "l2" or "cosine".
        """

        self.index: Any = None 
        self.k: int = k
        self.metric: str = metric
        self.radius: Optional[float] = radius
        self.res: Optional[Any] = res 


    def fit(self, data: torch.Tensor) -> None:
        """Creates index from provided vectors.

        Args:
            X: (num_vectors, dimensionality)
        """

        dimensions = data.shape[1]

        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(dimensions)
            if data.is_cuda:
                data = data.cpu()
            self.index.add(data)

        elif self.metric == "cosine":
            self.index = faiss.IndexFlatIP(dimensions)

            # Normalization.
            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            self.index.train(data)
            self.index.add(data)

        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def search(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds nearest neighbors.

        Args:
            X: (num_vectors, dimensionality)
        Returns:
            Distances and indices of the k nearest neighbors.
        """

        is_cuda = False
        device=None
        if data.is_cuda:
            device = data.get_device()
            is_cuda = True
            data = data.cpu()

        if self.metric == "l2":
            if self.radius is None:
                distances, indices = self.index.search(data, k=self.k)
            else:
                # Convert radii to float type for faster computation.
                distances, indices = self.index.range_search_with_radius(
                    data, float(self.radius)
                )

        elif self.metric == "cosine":

            # Normalize the query vectors.
            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            similarity, indices = self.index.search(data, k=self.k)

            # Cosine similarity to cosine distance.
            distances = 1.0 - similarity
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

        if is_cuda:
            distances = distances.to(device)
            indices = indices.to(device)

        return distances, indices

    def serialize_index(self) -> None:
        self.index = faiss.serialize_index(self.index)

    def deserialize_index(self) -> None:
        self.index = faiss.deserialize_index(self.index)
