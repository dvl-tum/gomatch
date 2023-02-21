from argparse import Namespace
from typing import Any, Dict, Mapping, Sequence, Union
import os

import numpy as np
import torch.utils.data as data
import yaml

from .data_processing import (
    load_scene_data,
    subsample_points_indices,
    compute_gt_2d3d_match,
    generate_assignment_mask,
    collect_covis_p3d_data,
)
from ..utils.geometry import points2d_to_bearing_vector

FEATURE_DIRS = {
    "sift": "SIFT1024",
    "superpoint": "SuperPoint_r4",
}


class BaseDataset(data.Dataset):
    default_config = dict(
        data_root="data/",
        dataset_conf="configs/datasets.yml",
        dataset="megadepth",
        p2d_type="sift",
        p3d_type="bvs",
        topk=1,
        npts=[10, 1024],
        outlier_rate=[0, 1],
        inls2d_thres=0.001,
        normalized_thres=True,
        merge_p3dm=True,
        random_topk=False,
    )

    def __init__(
        self, config: Union[Namespace, Mapping[str, Any]], split: str = "train"
    ) -> None:
        if isinstance(config, Namespace):
            config = vars(config)
        config = Namespace(**{**self.default_config, **config})
        self.topk = config.topk
        self.p2d_type = config.p2d_type
        self.p3d_type = config.p3d_type
        self.inls2d_thres = config.inls2d_thres
        self.normalized_thres = config.normalized_thres
        self.npts = config.npts
        self.outlier_rate = config.outlier_rate
        self.load_desc = True if self.p3d_type == "visdesc" else False
        self.merge_p3dm = config.merge_p3dm
        self.random_topk = config.random_topk

        # Load dataset configs
        self.split = split
        self.data_root = config.data_root
        self.dataset = config.dataset
        with open(config.dataset_conf, "r") as f:
            dataset_conf = Namespace(
                **yaml.load(f, Loader=yaml.FullLoader)[self.dataset]
            )
            self._load_scene_data(dataset_conf)

    def _load_scene_data(self, dataset_conf: Namespace) -> None:
        self.data_dir = data_dir = os.path.join(
            self.data_root, dataset_conf.data_processed_dir
        )
        scene3d_file = os.path.join(data_dir, "scene_points3d", f"{self.split}.npy")
        if not os.path.exists(scene3d_file):
            # Load pickle version
            scene3d_file = os.path.join(data_dir, "scene_points3d", f"{self.split}.pkl")

        data_file = os.path.join(data_dir, dataset_conf.data_file)
        self.feature_dir = os.path.join(
            data_dir, "desc_cache", FEATURE_DIRS[self.p2d_type]
        )
        scenes = dataset_conf.splits[self.split]
        self.n_scenes = len(scenes)
        data = load_scene_data(
            data_file,
            scenes=scenes,
            scene3d_file=scene3d_file,
            feature_dir=self.feature_dir,
            load_desc=self.load_desc,
        )
        self.sids, self.qids, self.ims, self.pts3d_data = data

    def _construct_data(
        self,
        query: Namespace,
        scene_pts3d: Sequence[np.ndarray],
        scene_ims: Mapping[int, Namespace],
    ) -> Dict[str, Any]:
        K, R, t, pts2d = query.K, query.R, query.t, query.kpts

        # Collect 3d data from topk retrival/covisible db images
        (pts3d, pts3dm, unmerge_mask, covis_ids) = collect_covis_p3d_data(
            query,
            self.topk,
            scene_pts3d,
            scene_ims,
            p3d_type=self.p3d_type,
            merge_pts3dm=self.merge_p3dm,
            random_topk=self.random_topk,
            npts=self.npts,
        )

        if self.outlier_rate == [0, 1] and self.merge_p3dm:
            # Point subsampling might be needed after merging
            if len(pts3d) > self.npts[1]:
                rids = np.random.choice(len(pts3d), self.npts[1], replace=False)
                pts3d = pts3d[rids]
                pts3dm = pts3dm[rids]
            elif len(pts3d) < self.npts[0]:
                pts2d = np.empty([0, 2])
                pts3d = np.empty([0, 3])
                pts3dm = np.empty([0, pts3dm.shape[1]])

        # Compute pesudo ground truth for 2d 3d matching
        i2ds, i3ds, o2ds, o3ds = compute_gt_2d3d_match(
            pts2d,
            pts3d,
            K,
            R,
            t,
            inls_thres=self.inls2d_thres,
            normalize=self.normalized_thres,
        )

        if self.outlier_rate == [0, 1]:
            # Generate assignment mask
            n2d, n3d = len(pts2d), len(pts3d)
            matches_bin = np.zeros((n3d + 1, n2d + 1), dtype=bool)
            matches_bin[i3ds, i2ds] = True
            matches_bin[o3ds, -1] = True
            matches_bin[-1, o2ds] = True
        elif self.merge_p3dm:
            # Control outliers for training and ablation
            pts2d_ids, pts3d_ids, matches = subsample_points_indices(
                i2ds, i3ds, o2ds, o3ds, self.outlier_rate, self.npts
            )
            pts2d = pts2d[pts2d_ids]
            pts3d = pts3d[pts3d_ids]
            pts3dm = pts3dm[pts3d_ids]

            # Generate assignment mask
            matches_bin = generate_assignment_mask(matches, len(pts3d))

        # Normalize 2d points
        pts2dm = pts2d_bvs = points2d_to_bearing_vector(pts2d, K, vec_dim=2)
        if self.load_desc:
            pts2dm = query.descs

        # Construct data
        data = dict(
            name=query.name,
            pts2d=pts2d_bvs.astype(np.float32),
            pts2d_pix=pts2d.astype(np.float32),
            pts2dm=pts2dm.astype(np.float32),
            pts3d=pts3d.astype(np.float32),
            pts3dm=pts3dm.astype(np.float32),
            matches_bin=matches_bin,
            R=R.astype(np.float32),
            t=t.astype(np.float32),
            K=K.astype(np.float32),
        )
        if not self.merge_p3dm:
            data["unmerge_mask"] = unmerge_mask
            data["covis_ids"] = covis_ids
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sid = self.sids[index]
        qid = self.qids[index]
        query = self.ims[sid][qid]

        # Load scene data
        scene_pts3d = self.pts3d_data[sid]  # pid: [xyz rgb err]
        scene_ims = self.ims[sid]

        # Generate kpts and labels for 2d-3d matching
        data = self._construct_data(query, scene_pts3d, scene_ims)
        return data

    def __len__(self) -> int:
        return len(self.qids)

    def __repr__(self) -> str:
        fmt_str = f"\nDataset:{self.dataset} split={self.split} "
        fmt_str += f"scenes:{self.n_scenes} queries: {len(self.qids)}\n"
        fmt_str += f"Data processed dir: {self.data_dir}\n"
        fmt_str += f"Settings=(\n"
        fmt_str += f"  topk={self.topk}, random_topk={self.random_topk}, orate={self.outlier_rate}, npt={self.npts},\n"
        fmt_str += f"  p2d_type={self.p2d_type}, p3d_type={self.p3d_type} load_desc={self.load_desc}, merge_p3dm={self.merge_p3dm}\n"
        fmt_str += f"  inls_thres={self.inls2d_thres} normalized_thres={self.normalized_thres}\n"
        fmt_str += f")\n"
        return fmt_str
