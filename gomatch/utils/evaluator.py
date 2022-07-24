from argparse import Namespace
from collections import defaultdict
import random
import time
from typing import Any, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .extract_matches import mutual_assignment
from .logger import get_logger
from .metrics import compute_metrics_sample, compute_metrics_batch, summarize_metrics
from .typing import Device, PathT, TensorOrArray
from .. import models
from ..data.datasets import BaseDataset
from ..data.data_processing import extract_covis_pts3d_ids

_logger = get_logger(level="INFO", name="evaluator")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sample_inputs(
    pts2d: TensorOrArray, pts3d: TensorOrArray, device: Device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(pts2d, np.ndarray):
        pts2d = torch.from_numpy(pts2d)
    if isinstance(pts3d, np.ndarray):
        pts3d = torch.from_numpy(pts3d)
    pts2d = pts2d.to(device)
    pts3d = pts3d.to(device)
    idx2d = torch.full((len(pts2d),), 0, device=device)
    idx3d = torch.full((len(pts3d),), 0, device=device)
    return pts2d, idx2d, pts3d, idx3d


class GenericModelWrapper(torch.nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        if config.matcher_class == "BPnPMatcher":
            self.matcher = models.BPnPMatcher()
        else:
            self.matcher = vars(models)[config.matcher_class](
                p3d_type=config.p3d_type,
                share_kp2d_enc=config.share_kp2d_enc,
                att_layers=config.att_layers,
            )

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        return self.matcher(pts2d, idx2d, pts3d, idx3d)


def load_matcher_from_ckpt(ckpt_path: PathT) -> Tuple[nn.Module, Namespace]:
    ckpt = torch.load(ckpt_path)
    config = Namespace(**ckpt["hyper_parameters"])
    model = GenericModelWrapper(config)
    _logger.info(f"Load model from {ckpt_path}")
    _logger.info(
        f"matcher={config.matcher_class} ep={ckpt['epoch']} step={ckpt['global_step']}"
    )

    # Load state dict
    load_state = model.load_state_dict(ckpt["state_dict"])
    _logger.info(f"Load state dict: {load_state}")

    # Get inner matcher
    matcher = model.matcher
    matcher.eval()
    _logger.info(
        f"Init matcher(p3d_type={config.p3d_type}, share_kp2d_enc={config.share_kp2d_enc}, "
        f"att_layers={config.att_layers})"
    )
    return matcher, config


class MatcherEvaluator:
    def __init__(
        self,
        vismatch: bool = False,
        ckpt_path: Optional[PathT] = None,
        sc_thres: float = 0.5,
        ransac_thres: float = 0.001,
        iterations_count: int = 1000,
        confidence: float = 0.99,
        oracle: bool = False,
    ) -> None:
        torch.set_grad_enabled(False)
        self.device = torch.device(
            "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
        )
        self.oracle = oracle
        self.sc_thres = sc_thres
        self.ransac_thres = ransac_thres
        self.iterations_count = iterations_count
        self.confidence = confidence
        self.metrics: defaultdict[str, Any] = defaultdict(list)
        self.list2device = lambda x: [v.to(self.device) for v in x]

        # Initialize model
        if self.oracle:
            self.matcher = None
            self.p3d_type = "coords"
        else:
            if vismatch:
                matcher: nn.Module = models.VisDescMatcher()
                self.p3d_type = "visdesc"
            else:
                assert ckpt_path is not None
                matcher, config = load_matcher_from_ckpt(ckpt_path)
                self.p3d_type = config.p3d_type
            self.matcher = matcher.to(self.device).eval()
        self.cls = isinstance(self.matcher, models.OTMatcherCls)

    def match_sample(
        self, pts2dm: TensorOrArray, pts3dm: TensorOrArray
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = prepare_sample_inputs(pts2dm, pts3dm, self.device)
        t0 = time.time()
        assert (
            self.matcher is not None
        ), "Unable to call `match_sample` if the evaluator is in oracle mode."
        preds = self.matcher(*inputs)
        self.metrics["match_time"].append(time.time() - t0)
        if self.cls:
            # Classification probs
            _, match_probs = preds
            match_scs = match_probs[0].cpu().data.numpy()
            match_est = (match_scs > self.sc_thres)[:-1, :-1]
        else:
            # Raw OT scores
            match_scs = preds[0].cpu().data.numpy()
            match_est = mutual_assignment(match_scs)[:-1, :-1]
        return match_est, match_scs

    def match_sample_separate_covis(
        self,
        pts2dm: TensorOrArray,
        pts3dm: np.ndarray,
        pts3d: np.ndarray,
        unmerge_mask: np.ndarray,
        covis_ids: np.ndarray,
        debug: bool = False,
    ) -> torch.Tensor:
        # TODO: debug parameter is not accessed

        # Extract pts3d ids for each covis pair inputs
        covis_pts3d_ids, covis_pts3dm_ids = extract_covis_pts3d_ids(
            covis_ids, unmerge_mask
        )

        # Initialize a single matching score matrix for k covis pairs
        matches_scores = np.zeros((len(pts3d), len(pts2dm)), dtype=float)

        # Match per query-covis pair
        for pts3d_ids, pts3dm_ids in zip(covis_pts3d_ids, covis_pts3dm_ids):
            ipts3dm = pts3dm[pts3dm_ids]
            imatches_est, iscores = self.match_sample(pts2dm, ipts3dm)

            # Parse matches
            i3ds_covis, i2ds = np.where(imatches_est)
            i3ds = pts3d_ids[i3ds_covis]
            assert (pts3d[i3ds] == pts3d[pts3d_ids][i3ds_covis]).all()

            # Keep the highest score
            matches_scores[i3ds, i2ds] = np.maximum(
                matches_scores[i3ds, i2ds], iscores[i3ds_covis, i2ds]
            )

        # Merged matches
        matches_est = matches_scores > 0
        return torch.from_numpy(matches_est).to(self.device)

    def eval_batch_merge_before_match(
        self, data: Mapping[str, torch.Tensor], debug: bool = False
    ) -> None:
        """Perform matching on 3D data merged from k covis views.
        In this case, 3D data is already merged within the data loader.
        The merged data has the same format as loading data from a single
        covis view, i.e., k=1.
        Therefore, this function matches batch data loaded with k=1 or merged k>1.
        """
        if not self.oracle:
            # Load data for matching
            inputs = self.list2device(
                [data["pts2dm"], data["idx2d"], data["pts3dm"], data["idx3d"]]
            )

            # Matching
            t0 = time.time()
            assert (
                self.matcher is not None
            ), "Unable to call `eval_batch_merge_before_match` if the evaluator is in oracle mode."
            preds = self.matcher(*inputs)
            self.metrics["match_time"].append(time.time() - t0)
        else:
            preds = None

        # Compute metrics
        compute_metrics_batch(
            self.metrics,
            data,
            preds,
            cls=self.cls,
            sc_thres=self.sc_thres,
            ransac_thres=self.ransac_thres,
            is_test=True,
            oracle=self.oracle,
            debug=debug,
            iterations_count=self.iterations_count,
            confidence=self.confidence,
        )

    def eval_batch_merge_after_match(
        self, data: Mapping[str, Any], debug: bool = False
    ) -> None:
        bids = torch.unique_consecutive(data["idx2d"])
        i = 0
        for bid in bids:
            mask2d = data["idx2d"] == bid
            mask3d = data["idx3d"] == bid
            mask3dm = data["idx3dm"] == bid
            n2d = mask2d.sum()
            n3d = mask3d.sum()
            total = (n2d + 1) * (n3d + 1)
            matches_gt = data["matches_bin"][i : i + total].view(n3d + 1, n2d + 1)
            i += total
            if self.oracle:
                matches_est = matches_gt
                pts3d = data["pts3d"][mask3d]
            else:
                # Load data for matching
                pts2dm = data["pts2dm"][mask2d]
                pts3d = data["pts3d"][mask3d]
                pts3dm = data["pts3dm"][mask3dm]
                unmerge_mask = data["unmerge_mask"][bid]
                covis_ids = data["covis_ids"][bid]

                # Estimate matches from k query-covis pairs
                matches_est = self.match_sample_separate_covis(
                    pts2dm, pts3dm, pts3d, unmerge_mask, covis_ids
                )

            # Load data for evaluation
            pts2d = data["pts2d"][mask2d]
            pts2d_pix = data["pts2d_pix"][mask2d]
            R_gt = data["R"][bid]
            t_gt = data["t"][bid]
            K = data["K"][bid]

            # Compute metrics per sample
            compute_metrics_sample(
                self.metrics,
                matches_est,
                matches_gt,
                pts2d,
                pts2d_pix,
                pts3d,
                R_gt,
                t_gt,
                K,
                ransac_thres=self.ransac_thres,
                is_test=True,
                print_out=debug,
                iterations_count=self.iterations_count,
                confidence=self.confidence,
            )

    def eval_data_loader(self, data_loader: DataLoader, debug: bool = False) -> None:
        seed_everything(933)
        self.clear_metrics()

        dataset = cast(BaseDataset, data_loader.dataset)
        self.metrics["n_queries"] = len(dataset)
        merge_before_match = dataset.merge_p3dm
        _logger.info(f">>>Start evaluation..")
        _logger.info(dataset)
        _logger.info(
            f"Model: {self.matcher.__class__.__name__} sc_thres={self.sc_thres} ransac_thres={self.ransac_thres} merge_before_match={merge_before_match}"
        )
        t0 = time.time()
        for i, data in enumerate(tqdm(data_loader)):
            if merge_before_match:
                self.eval_batch_merge_before_match(data, debug=debug)
            else:
                self.eval_batch_merge_after_match(data, debug=debug)
            if debug and i >= 5:
                break
        self.summarize_eval()
        _logger.info(f"Evaluation finished, total runtime: {(time.time()-t0):.2f}s.")

    def clear_metrics(self) -> None:
        self.metrics = defaultdict(list)

    def summarize_eval(self) -> None:
        summarize_metrics(self.metrics)
