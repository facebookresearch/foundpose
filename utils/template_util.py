#!/usr/bin/env python3

from typing import Optional, Tuple

import torch

from utils import knn_util, repre_util, logging, misc

logger: logging.Logger = logging.get_logger()


def find_nearest_object_features(
    query_features: torch.Tensor,
    knn_index: knn_util.KNN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find the nearest reference feature for each query feature.
    nn_dists, nn_ids = knn_index.search(query_features)

    knn_k = nn_dists.shape[1]

    # Keep only the required k nearest neighbors.
    nn_dists = nn_dists[:, :knn_k]
    nn_ids = nn_ids[:, :knn_k]

    # The distances returned by faiss are squared.
    nn_dists = torch.sqrt(nn_dists)

    return nn_ids, nn_dists

def calc_tfidf(
    feature_word_ids: torch.Tensor,
    feature_word_dists: torch.Tensor,
    word_idfs: torch.Tensor,
    soft_assignment: bool = True,
    soft_sigma_squared: float = 100.0,
) -> torch.Tensor:
    """Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf"""

    device = feature_word_ids.device

    # Calculate soft-assignment weights, as in:
    # "Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases"
    if soft_assignment:
        word_weights = torch.exp(
            -torch.square(feature_word_dists) / (2.0 * soft_sigma_squared)
        )
    else:
        word_weights = torch.ones_like(feature_word_dists)

    # Normalize the weights such as they sum up to 1 for each query.
    word_weights = torch.nn.functional.normalize(word_weights, p=2, dim=1).reshape(-1)

    # Calculate term frequencies.
    # tf = word_weights  # https://www.cs.cmu.edu/~16385/s17/Slides/8.2_Bag_of_Visual_Words.pdf
    tf = word_weights / feature_word_ids.shape[0]  # From "Lost in Quantization".

    # Calculate inverse document frequencies.
    feature_word_ids_flat = feature_word_ids.reshape(-1)
    idf = word_idfs[feature_word_ids_flat]

    # Calculate tfidf values.
    tfidf = torch.multiply(tf, idf)

    # Construct the tfidf descriptor.
    num_words = word_idfs.shape[0]
    tfidf_desc = torch.zeros(
        num_words, dtype=word_weights.dtype, device=device
    ).scatter_add_(dim=0, index=feature_word_ids_flat.to(torch.int64), src=tfidf)

    return tfidf_desc


def calc_tfidf_descriptors(
    feat_vectors: torch.Tensor,
    feat_to_word_ids: torch.Tensor,
    feat_to_template_ids: torch.Tensor,
    feat_words: torch.Tensor,
    num_templates: int,
    tfidf_knn_k: int,
    tfidf_soft_assign: bool,
    tfidf_soft_sigma_squared: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate tf-idf descriptors.

    For each visual word i (i.e. cluster), idf is defined as log(N / N_i), where N is
    the number of images and N_i is the number of images in which visual word i appears.

    Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf
    """

    device = feat_words.device.type

    # Calculate the idf terms (inverted document frequency).
    word_occurances = torch.zeros(len(feat_words), dtype=torch.int64, device=device)
    for template_id in range(num_templates):
        mask = feat_to_template_ids == template_id
        unique_word_ids = torch.unique(feat_to_word_ids[mask])
        word_occurances[unique_word_ids] += 1
    word_idfs = torch.log(
        torch.as_tensor(float(num_templates)) / word_occurances.to(torch.float32)
    )

    # Build a KNN index for the visual words.
    feat_knn_index = knn_util.KNN(k=tfidf_knn_k, metric="l2")
    feat_knn_index.fit(feat_words.cpu())

    # Calculate the tf-idf descriptor for each template.
    tfidf_descs = []
    for template_id in range(num_templates):
        tpl_mask = feat_to_template_ids == template_id
        word_dists, word_ids = feat_knn_index.search(feat_vectors[tpl_mask])
        tfidf = calc_tfidf(
            feature_word_ids=word_ids,
            feature_word_dists=word_dists,
            word_idfs=word_idfs,
            soft_assignment=tfidf_soft_assign,
            soft_sigma_squared=tfidf_soft_sigma_squared,
        )
        tfidf_descs.append(tfidf)
    tfidf_descs = torch.stack(tfidf_descs, dim=0)

    return tfidf_descs, word_idfs


def tfidf_matching(
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    top_n_templates: int,
    visual_words_knn_index: knn_util.KNN,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf"""

    if (
        object_repre.template_desc_opts is None
        or object_repre.template_desc_opts.desc_type != "tfidf"
    ):
        raise ValueError("Template descriptors need to be tfidf.")

    timer = misc.Timer(enabled=debug)
    timer.start()

    # For each query vector, find the nearest visual words.
    word_ids, word_dists = find_nearest_object_features(
        query_features=query_features,
        knn_index=visual_words_knn_index,
    )

    timer.elapsed("Time for KNN search")

    # Calculate tfidf vector of the query image.
    assert object_repre.feat_cluster_idfs is not None
    assert object_repre.template_desc_opts is not None
    query_tfidf = calc_tfidf(
        feature_word_ids=word_ids,
        feature_word_dists=word_dists,
        word_idfs=object_repre.feat_cluster_idfs,
        soft_assignment=object_repre.template_desc_opts.tfidf_soft_assign,
        soft_sigma_squared=object_repre.template_desc_opts.tfidf_soft_sigma_squared,
    )

    # Calculate cosine similarity between the query descriptor and the template descriptors.
    assert object_repre.template_descs is not None
    num_templates = object_repre.template_descs.shape[0]
    assert object_repre.template_descs is not None
    match_feat_cos_sims = torch.nn.functional.cosine_similarity(
        object_repre.template_descs, query_tfidf.tile(num_templates, 1)
    )

    # Select templates with the highest cosine similarity.
    template_scores, template_ids = torch.topk(
        match_feat_cos_sims, k=top_n_templates, sorted=True
    )

    return template_ids, template_scores


def template_matching(
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    top_n_templates: int,
    matching_type: str,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieves N most similar templates to the query image."""

    if matching_type == "tfidf":
        assert visual_words_knn_index is not None
        template_ids, template_scores = tfidf_matching(
            query_features=query_features,
            object_repre=object_repre,
            top_n_templates=top_n_templates,
            visual_words_knn_index=visual_words_knn_index,
        )

    else:
        raise ValueError(f"Unknown matching type '{matching_type}'.")

    logger.info(f"Matched templates: {list(misc.tensor_to_array(template_ids))}")

    return template_ids, template_scores
