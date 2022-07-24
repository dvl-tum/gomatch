from typing import Optional, Tuple

import numpy as np
from scipy.spatial.ckdtree import cKDTree
import torch

from .typing import TensorOrArray


def mutual_assignment(match_scs: TensorOrArray) -> np.ndarray:
    if isinstance(match_scs, torch.Tensor):
        match_scs = match_scs.cpu().data.numpy()

    # match_scs: (n3d+1, n2d+1)
    scores = match_scs[:-1, :-1]
    n3d, n2d = scores.shape

    # Extract matches
    nn12 = scores.argmax(1)
    nn21 = scores.argmax(0)
    m12 = np.dstack([np.arange(n3d), nn12]).squeeze()
    m21 = np.dstack([nn21, np.arange(n2d)]).squeeze()

    # Mutually matched
    match_ids = np.concatenate([m12, m21], axis=0)
    _, ids, counts = np.unique(match_ids, axis=0, return_index=True, return_counts=True)
    match_ids_ = match_ids[ids[counts > 1]]

    # To avoid pose failure cases as much as possible
    if len(match_ids_) <= 4:
        match_ids_ = match_ids[ids]

    # Construct assignment mask for metric calculation
    assign_mask = np.zeros_like(match_scs)
    if len(match_ids_.shape) == 2:  # In cases we have no mutual matches
        ids1, ids2 = match_ids_[:, 0], match_ids_[:, 1]
        assign_mask[ids1, ids2] = 1
    dust_col = 1 - assign_mask.sum(1)
    dust_row = 1 - assign_mask.sum(0)
    assign_mask[:, -1] = dust_col
    assign_mask[-1, :] = dust_row
    assign_mask[-1, -1] = 0
    assign_mask = assign_mask.astype(bool)
    return assign_mask


def cal_mutual_nn_dists_kdtrees(
    nn12: np.ndarray,
    nn21: np.ndarray,
    dist12: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Mutual nearest matches wrt min distances
    ids1 = np.arange(0, len(nn12))
    mutual_mask = ids1 == nn21[nn12]
    ids1 = ids1[mutual_mask]
    ids2 = nn12[mutual_mask]
    match_ids = np.stack([ids1, ids2]).T
    match_dists = dist12[mutual_mask]
    if threshold:
        thres_mask = match_dists < threshold
        match_ids = match_ids[thres_mask]
        match_dists = match_dists[thres_mask]
    return match_ids, match_dists


def align_points2d(
    pts1: np.ndarray, pts2: np.ndarray, dist_thres: Optional[float] = None
) -> np.ndarray:
    # Measure pair-wise distances
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    dist12, nn12 = tree2.query(pts1)
    dist21, nn21 = tree1.query(pts2)

    # Define inliers with the nearest mutual matches
    aligned_ids, _ = cal_mutual_nn_dists_kdtrees(
        nn12, nn21, dist12, threshold=dist_thres
    )
    return aligned_ids
