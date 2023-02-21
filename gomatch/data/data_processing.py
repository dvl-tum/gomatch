from argparse import Namespace
import os
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pickle
import numpy as np
import torch


from ..utils.extract_matches import align_points2d
from ..utils.geometry import (
    points2d_to_bearing_vector,
    project_points3d,
    project3d_normalized,
)
from ..utils.logger import get_logger
from ..utils.typing import PathT, TensorOrArrayOrList

_logger = get_logger(level="INFO", name="data_process")


def load_scene_data(
    data_file: PathT,
    scenes: Collection[str],
    scene3d_file: PathT,
    feature_dir: PathT,
    load_desc: bool = False,
) -> Tuple[List, List, Dict[str, Any], Dict[str, Any]]:
    _logger.info(f"Loading data file from {data_file}")
    data_dict = np.load(data_file, allow_pickle=True).item()

    # Load entire 3d point data
    _logger.info(f"Loading scene 3D points from {scene3d_file} ...")
    if scene3d_file.split('.')[-1] == 'pkl':
        with open(scene3d_file, 'rb') as handle:
            pts3d_data = pickle.load(handle)
    else:
        pts3d_data = np.load(scene3d_file, allow_pickle=True).item()
    _logger.info(f"Done with 3D data loading.")

    # Load all query ids, scene ids and image data
    sids = []
    qids = []
    ims = {}
    num_pts3d = []
    _logger.info(
        f"Fetching scene data for: {len(scenes)} scenes\nFeature dir:{feature_dir}"
    )
    for sid in scenes:
        if sid not in data_dict:
            continue

        # Load pre-computed 2d feature
        scene_features = np.load(
            os.path.join(feature_dir, sid + ".npy"), allow_pickle=True
        ).item()

        # Extract scene data
        scene_pts3d = pts3d_data[sid]
        scene_ims = data_dict[sid]["ims"]
        scene_qids = data_dict[sid]["qids"]

        # Store data
        qids += scene_qids
        sids += [sid] * len(scene_qids)
        ims[sid] = scene_ims

        # Assign image with features
        for qid, im in scene_ims.items():
            pts2d = scene_features[qid]["kpts"]
            if isinstance(pts2d, torch.Tensor):
                pts2d = pts2d.cpu().data.numpy()
            pts2d, pts2d_uids = np.unique(pts2d, axis=0, return_index=True)
            im.kpts = pts2d
            if load_desc:
                # This is used by vismatch
                descs = scene_features[qid]["descs"][pts2d_uids]
                if isinstance(descs, torch.Tensor):
                    descs = descs.cpu().data.numpy()
                im.descs = descs
            if im.pts3d is None:
                continue

            # Match gt pts3d and detected pts2d
            pts3d_ids = im.pts3d
            pts3d = np.stack([scene_pts3d[pid][:3] for pid in pts3d_ids])
            pts3d_proj, valid = project_points3d(im.K, im.R, im.t, pts3d)
            aligned_ids = align_points2d(pts2d, pts3d_proj[valid])
            i2ds, i3ds = aligned_ids[:, 0], aligned_ids[:, 1]

            # Keep only matched 3d points
            im.pts3d = pts3d_ids[valid][i3ds]
            num_pts3d.append(len(im.pts3d))

            # Later nned it to assign desc to 3d points
            im.aligned_i2ds = i2ds

    _logger.info(
        f"Finished loading scenes: {len(pts3d_data)}, queries: {len(qids)} im.pts3d:{int(np.mean(num_pts3d))} #ims(pts3d <20)={(np.array(num_pts3d) < 20).sum()}"
    )
    return sids, qids, ims, pts3d_data


def collect_covis_p3d_data(
    query: Namespace,
    topk: int,
    pts_data: Sequence[np.ndarray],
    ims_data: Mapping[int, Namespace],
    p3d_type: str = "bvs",
    npts: Tuple[int, int] = (0, 1024),
    merge_pts3dm: bool = False,
    random_topk: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npts_min, npts_max = npts
    topk_num = min(topk, len(query.topk))
    covis_im_ids = (
        np.random.choice(query.topk, topk_num, replace=False)
        if random_topk
        else query.topk[:topk_num]
    )

    # Collect 3d points from covis-k db ims
    db_pids = []
    pts3d = []
    pts3dm_ = []
    covis_ids_all = []
    for i, idx in enumerate(covis_im_ids):
        db_im = ims_data[idx]
        db_pids_i = db_im.pts3d
        npts3d_i = len(db_pids_i)
        if npts3d_i == 0:
            continue

        if not merge_pts3dm:
            # It is optimal to do the point subsampling here.
            if npts3d_i < npts_min:
                continue
            if npts3d_i > npts_max:
                db_pids_i = np.random.choice(db_pids_i, npts_max, replace=False)

        # Collect data
        covis_ids_all += [i] * len(db_pids_i)
        db_pids.append(db_pids_i)
        pts3d_i = np.array([pts_data[i][:3] for i in db_pids_i])
        pts3d.append(pts3d_i)
        if p3d_type == "bvs":
            pts3dm_.append(project3d_normalized(db_im.R, db_im.t, pts3d_i))
        elif p3d_type == "visdesc":
            pts3dm_.append(db_im.descs[db_im.aligned_i2ds])
        elif p3d_type == "coords":
            pts3dm_.append(pts3d_i)
    pts3d = np.concatenate(pts3d)
    pts3dm = np.concatenate(pts3dm_)
    db_pids = np.concatenate(db_pids)
    covis_ids = np.array(covis_ids_all)

    # Merge k covis points
    if topk == 1:
        merge_mask = unmerge_mask = np.arange(len(pts3d))
    else:
        _, merge_mask, unmerge_mask = np.unique(
            db_pids, return_index=True, return_inverse=True
        )
    pts3d_merged = pts3d[merge_mask]
    pts3dm_merged = pts3dm[merge_mask] if merge_pts3dm else pts3dm
    return pts3d_merged, pts3dm_merged, unmerge_mask, covis_ids


def extract_covis_pts3d_ids(
    covis_ids: np.ndarray, unmerge_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    covis_pts3d_ids = []
    covis_pts3dm_ids = []
    start_idx = 0
    covis_id_vals = np.unique(covis_ids)
    for i in covis_id_vals:
        mask = covis_ids == i
        npts = mask.sum()
        end_idx = start_idx + npts
        pts3d_ids = unmerge_mask[start_idx:end_idx]
        pts3dm_ids = np.arange(start_idx, end_idx)
        covis_pts3d_ids.append(pts3d_ids)
        covis_pts3dm_ids.append(pts3dm_ids)
        start_idx = end_idx
    return covis_pts3d_ids, covis_pts3dm_ids


def align_2d3d_points_normalized(
    pts2d: np.ndarray,
    pts3d_proj: np.ndarray,
    K: np.ndarray,
    dist_thres: Optional[float],
    radial: Optional[float] = None,
) -> np.ndarray:
    # Convert pts2d to normalized coordinates
    pts2d_normed = points2d_to_bearing_vector(pts2d, K, vec_dim=2, radial=radial)
    pts3d_proj_normed = points2d_to_bearing_vector(
        pts3d_proj, K, vec_dim=2, radial=radial
    )
    aligned_ids = align_points2d(pts2d_normed, pts3d_proj_normed, dist_thres=dist_thres)
    return aligned_ids


def compute_gt_2d3d_match(
    pts2d: np.ndarray,
    pts3d: TensorOrArrayOrList,
    K: np.ndarray,
    R: TensorOrArrayOrList,
    t: TensorOrArrayOrList,
    inls_thres: Optional[float] = 1,
    normalize: bool = False,
    radial: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Project 3d keypoints onto image plane
    pts2d_proj, valid = project_points3d(K, R, t, pts3d, radial=radial)

    # Align 2d points with 3d projections
    if normalize:
        # NN matching based on normalized distances
        matches = align_2d3d_points_normalized(
            pts2d, pts2d_proj[valid], K, dist_thres=inls_thres, radial=radial
        )
    else:
        # NN matching based on image pixel distances
        matches = align_points2d(pts2d, pts2d_proj[valid], dist_thres=inls_thres)
    i3d_map = np.where(valid)[0]
    i2ds, i3ds = matches[:, 0], i3d_map[matches[:, 1]]

    # Everything not inliers as outliers
    n2d, n3d = len(pts2d), len(valid)
    inls_mask2d = np.zeros(n2d, dtype=bool)
    inls_mask2d[i2ds] = True
    inls_mask3d = np.zeros(n3d, dtype=bool)
    inls_mask3d[i3ds] = True
    o2ds = np.where(~inls_mask2d)[0]
    o3ds = np.where(~inls_mask3d)[0]
    return i2ds, i3ds, o2ds, o3ds


def enforce_outlier_rate_and_npts(
    ni: int,
    no2d: int,
    no3d: int,
    orate: Union[float, Tuple[float, float]],
    npts: Union[int, Tuple[int, int]] = -1,
) -> Tuple[int, int, int]:
    if isinstance(orate, float):
        orate = (orate, orate)
    ormin, ormax = orate
    if isinstance(npts, int):
        npts = (0, npts)
    npts_min, npts_max = npts

    # Compute inlier and outlier numbers to fulfil the range
    if min(no2d / (ni + no2d), no3d / (ni + no3d)) < ormin:
        # Not enough outliers, cut inliers
        ni = int(min(no2d, no3d) * (1 - ormin) / ormin)
    nomax = (
        int(ni * ormax / (1 - ormax)) if ormax >= 0 and ormax < 1 else max(no2d, no3d)
    )
    no2d = min(no2d, nomax)
    no3d = min(no3d, nomax)

    if npts_max > 0 and max(no2d, no3d) + ni > npts_max:
        # Rescale up to maximum points
        scale = npts_max / (ni + max(no2d, no3d))
        ni = int(scale * ni)
        no2d = int(scale * no2d)
        no3d = int(scale * no3d)

    if min(no2d, no3d) + ni < npts_min:
        # Set to empty to ignore those samples
        no2d = no3d = ni = 0
    return ni, no2d, no3d


def subsample_points_indices(
    i2ds: np.ndarray,
    i3ds: np.ndarray,
    o2ds: np.ndarray,
    o3ds: np.ndarray,
    orate: Union[float, Tuple[float, float]],
    npts: Union[int, Tuple[int, int]] = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Supsample data points to fulfil outlier rate and max points
    ni, no2d, no3d = len(i2ds), len(o2ds), len(o3ds)
    ni_, no2d_, no3d_ = enforce_outlier_rate_and_npts(ni, no2d, no3d, orate, npts)

    # Sampling
    iid = np.random.choice(ni, ni_, replace=False)
    oid2d = np.random.choice(no2d, no2d_, replace=False)
    oid3d = np.random.choice(no3d, no3d_, replace=False)
    i2ds_, i3ds_ = i2ds[iid], i3ds[iid]
    o2ds_, o3ds_ = o2ds[oid2d], o3ds[oid3d]

    # Construct data
    pts2d_ids = np.concatenate([i2ds_, o2ds_])
    pts3d_ids = np.concatenate([i3ds_, o3ds_])
    matches = np.concatenate([np.arange(ni_), np.full(no2d_, -1)])  # Map from 2d to 3d

    # Randomly shuffle orders to avoid system bias in learning
    idx3d = np.random.permutation(np.arange(len(pts3d_ids)))
    idx3d_inv = np.argsort(idx3d)
    pts3d_ids = pts3d_ids[idx3d]
    matches[:ni_] = idx3d_inv[matches[:ni_]]
    idx2d = np.random.permutation(np.arange(len(pts2d_ids)))
    pts2d_ids = pts2d_ids[idx2d]
    matches = matches[idx2d]
    return pts2d_ids, pts3d_ids, matches


def generate_assignment_mask(matches: np.ndarray, n3d: int) -> np.ndarray:
    n2d = len(matches)
    mask = np.zeros((n3d + 1, n2d + 1), dtype=bool)

    # Fill matches
    matched = matches > -1
    match_2d = np.where(matched)[0]
    match_3d = matches[matched]
    mask[match_3d, match_2d] = True

    # Fill dustbines
    mask[-1, np.where(~matched)[0]] = True
    mask[:-1, -1] = ~np.logical_or.reduce(mask[:-1].T)
    return mask
