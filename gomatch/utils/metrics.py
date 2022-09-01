from argparse import Namespace
from math import pi
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch

from .extract_matches import mutual_assignment
from .geometry import estimate_pose, project3d_normalized, project_points3d
from .logger import get_logger
from .typing import TensorOrArray

_logger = get_logger(level="INFO", name="metrics")

# TODO: fix the return interface
def check_data_hist(
    data_list: Collection[np.ndarray],
    bins: Sequence[float],
    tag: str = "",
    return_hist: bool = False,
) -> Union[str, np.ndarray, Tuple[str, None], Tuple[np.ndarray, str]]:
    if not data_list:
        if return_hist:
            return "", None
        return ""
    hists = []
    means = []
    Ns = []
    for data in data_list:
        N = len(data)
        Ns.append(N)
        if N == 0:
            continue
        counts = np.histogram(data, bins)[0]
        hists.append(counts / N)
        means.append(np.mean(data))

    hist_print = f"{tag} Sample/N(mean/max/min)={len(data_list)}/{np.mean(Ns):.0f}/{np.max(Ns):.0f}/{np.min(Ns):.0f}\n"
    hist_print += f"Ratios(%): mean={np.mean(means):.2f}"
    mean_hists = np.mean(hists, axis=0)
    for val, low, high in zip(mean_hists, bins[0:-1], bins[1::]):
        hist_print += " [{},{})={:.2f}".format(low, high, 100 * val)
    if return_hist:
        return mean_hists, hist_print
    return hist_print


def cal_error_auc(errors: Sequence[float], thresholds: Collection[float]) -> np.ndarray:
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors_ = np.append([0.0], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs: List[float] = []
    for thres in thresholds:
        last_index = cast(int, np.searchsorted(errors_, thres))
        rcs_ = np.append(recalls[:last_index], recalls[last_index - 1])
        errs_ = np.append(errors_[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs)


def reprojection_err_normalized(
    matches_est: torch.Tensor,
    pts2d_bvs: torch.Tensor,
    pts3d: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
) -> torch.Tensor:
    i3d, i2d = torch.where(matches_est)

    # Meaure reprojection err on normalized coordinates
    kps2d_proj = project3d_normalized(R_gt, t_gt, pts3d[i3d])
    match_dists = (pts2d_bvs[i2d, :2] - kps2d_proj).norm(dim=1)
    return match_dists


def reprojection_err(
    matches_est: torch.Tensor,
    pts2d_pix: torch.Tensor,
    pts3d: torch.Tensor,
    K: TensorOrArray,
    R_gt: TensorOrArray,
    t_gt: TensorOrArray,
) -> torch.Tensor:
    i3d, i2d = torch.where(matches_est)

    # Meaure reprojection err on image coordinates
    kps2d_proj, _ = project_points3d(K, R_gt, t_gt, pts3d[i3d])
    match_dists = (pts2d_pix[i2d] - pts2d_pix.new_tensor(kps2d_proj)).norm(dim=1)
    return match_dists


def pose_err(
    R: TensorOrArray, R_gt: TensorOrArray, t: TensorOrArray, t_gt: TensorOrArray
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(R_gt, np.ndarray):
        R_gt = torch.from_numpy(R_gt).to(torch.float32)
    if isinstance(t_gt, np.ndarray):
        t_gt = torch.from_numpy(t_gt).to(torch.float32)
    R_err = (
        torch.clip(0.5 * (torch.sum(R_gt * R_gt.new_tensor(R)) - 1), -1, 1).acos()
        * 180.0
        / pi
    )
    t_err = torch.norm(t_gt.new_tensor(t) - t_gt)
    return R_err, t_err


def io_metric(
    in_est: torch.Tensor, in_gt: torch.Tensor, all_metrics: bool = False
) -> Dict[str, torch.Tensor]:
    # Inlier/outlier classification metrics
    tp = torch.sum(in_est & in_gt)
    pgt = torch.sum(in_gt)
    pest = torch.sum(in_est)
    recall = tp / pgt if pgt > 0.0 else pgt.float()
    precision = tp / pest if pest > 0.0 else pest.float()
    metrics = dict(recall=recall, precision=precision)

    if all_metrics:
        ngt = torch.sum(~in_gt)
        tn = torch.sum(~in_gt & ~in_est)
        specifity = tn / ngt if ngt > 0.0 else ngt.float()
        accuracy = (tp + tn) / (pgt + ngt)
        metrics.update(dict(specifity=specifity, accuracy=accuracy))
    return metrics


def compute_metrics_sample(
    metrics: Mapping[str, List],
    matches_est: torch.Tensor,
    matches_gt: torch.Tensor,
    pts2d: torch.Tensor,
    pts2d_pix: torch.Tensor,
    pts3d: torch.Tensor,
    R_gt: TensorOrArray,
    t_gt: TensorOrArray,
    K: torch.Tensor,
    ransac_thres: float = 0.001,
    is_test: bool = False,
    print_out: bool = True,
    iterations_count: int = 1000,
    confidence: float = 0.99,
) -> None:
    n2d, n3d = len(pts2d), len(pts3d)
    if is_test:
        metrics["n2d"].append(n2d)
        metrics["n3d"].append(n3d)

    # Uniform assignment format
    matches_est_ = matches_est[:n3d, :n2d]
    matches_gt_ = matches_gt[:n3d, :n2d]

    # Pose estimation
    i3d, i2d = torch.where(matches_est_)
    pose_res = estimate_pose(
        pts2d[i2d, :2],
        pts3d[i3d],
        ransac_thres=ransac_thres,
        iterations_count=iterations_count,
        confidence=confidence,
    )
    if not pose_res:
        R_err: Union[float, torch.Tensor] = -1.0
        t_err: Union[float, torch.Tensor] = -1.0
        inliers: Union[List, np.ndarray] = []
    else:
        R, t, inliers = pose_res
        R_err, t_err = pose_err(R, R_gt, t, t_gt)
    metrics["R_err"].append(R_err)
    metrics["t_err"].append(t_err)
    metrics["n_matches"].append(len(i3d))

    if pose_res:
        # Reproject gt 3D points with estimated pose
        reproj_errs = (
            reprojection_err(matches_gt_, pts2d_pix, pts3d, K, R, t).cpu().data.numpy()
        )
    else:
        reproj_errs = np.empty(0)

    if len(reproj_errs) == 0:
        if is_test:
            reproj_errs = np.array([float("inf")])
        else:
            reproj_errs = np.array([-1.0])
    reproj_errs = reproj_errs if is_test else reproj_errs.mean()
    metrics["reproj_gt3d_estpose"].append(reproj_errs)

    if print_out:
        _logger.info(
            f"i3d={len(i3d)} R_err={R_err:.2f} t_err={t_err:.2f} inls={len(inliers)} "
            f"Npts={len(pts2d)}/{len(pts3d)} n_matches={len(i3d)}"
        )


def compute_metrics_batch(
    metrics: MutableMapping[str, Any],
    data: Mapping[str, torch.Tensor],
    preds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    cls: bool = False,
    sc_thres: float = -1,
    ransac_thres: float = 0.001,
    iterations_count: int = 1000,
    confidence: float = 0.99,
    is_test: bool = False,
    oracle: bool = False,
    debug: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    bids = torch.unique_consecutive(data["idx2d"])
    i = 0
    if cls:
        ot_scores_b, match_probs_b = preds
    else:
        ot_scores_b = preds
    for bid in bids:
        mask2d = data["idx2d"] == bid
        mask3d = data["idx3d"] == bid
        n2d = mask2d.sum()
        n3d = mask3d.sum()
        total = (n2d + 1) * (n3d + 1)
        matches_gt = data["matches_bin"][i : i + total].view(n3d + 1, n2d + 1)
        i += total

        if oracle:
            matches_est = matches_gt
        else:
            matches_est = torch.zeros_like(ot_scores_b[bid]).bool()
            if cls:
                # Compute matches based on cls scores
                i3ds: TensorOrArray
                i2ds: TensorOrArray
                i3ds, i2ds = torch.where(match_probs_b[bid] > sc_thres)
            else:
                # Compute matches based on OT scores
                i3ds, i2ds = np.where(mutual_assignment(ot_scores_b[bid]))
            matches_est[i3ds, i2ds] = True

        # Load Points
        pts2d = data["pts2d"][mask2d]
        pts2d_pix = data["pts2d_pix"][mask2d]
        pts3d = data["pts3d"][mask3d]
        R_gt = data["R"][bid]
        t_gt = data["t"][bid]
        K = data["K"][bid]

        compute_metrics_sample(
            metrics,
            matches_est,
            matches_gt,
            pts2d,
            pts2d_pix,
            pts3d,
            R_gt,
            t_gt,
            K,
            ransac_thres=ransac_thres,
            iterations_count=iterations_count,
            confidence=confidence,
            is_test=is_test,
            print_out=debug,
        )
    if is_test:
        return None

    # Reduce metrics
    raw_pose_errs = dict()
    for k, v in metrics.items():
        v = torch.tensor(v, dtype=torch.float)
        if k in ["R_err", "t_err"]:
            raw_pose_errs[k] = v
        v = v[v > -1]
        metrics[k] = torch.mean(v) if len(v) > 0 else -1.0
    return raw_pose_errs


# TODO: revise the return interface
def summarize_metrics(
    metrics: Union[Dict[str, Union[int, np.ndarray]], Namespace],
    auc_thresholds: Collection[float] = (1, 5, 10, 15, 50, 100, 500),
    pose_thresholds: Iterable[Tuple[float, float]] = (
        (0.05, 5),
        (0.25, 2),
        (0.5, 5),
        (1.0, 10),
    ),
    return_qs: bool = False,
) -> Union[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
    np.set_printoptions(precision=2)

    # Convert to namespace for easy reference
    if isinstance(metrics, dict):
        metrics = Namespace(**metrics)
    metrics.t_err = np.array(metrics.t_err, dtype=np.float32)
    metrics.R_err = np.array(metrics.R_err, dtype=np.float32)
    n_queries = metrics.n_queries
    pose_mask = (metrics.t_err > -1) & (metrics.R_err > -1)
    failed = int(len(metrics.t_err) - pose_mask.sum())

    # Query statis
    _logger.info(
        f"Query total={n_queries} "
        f"evaluated={len(metrics.t_err)} "
        f"failed={failed}"
    )

    # Match statis
    _logger.info(
        f"Mean n2d={np.mean(metrics.n2d):.0f} "
        f"n3d={np.mean(metrics.n3d):.0f} "
        f"n_matches={np.mean(metrics.n_matches):.0f}"
    )

    # Pose errors
    quantile_ratios = torch.tensor([0.25, 0.5, 0.75])
    t_qs = torch.quantile(torch.from_numpy(metrics.t_err[pose_mask]), quantile_ratios)
    R_qs = torch.quantile(torch.from_numpy(metrics.R_err[pose_mask]), quantile_ratios)
    _logger.info(
        f"R_qs={R_qs.data.numpy()} t_qs={t_qs.data.numpy()}m/{t_qs.data.numpy()*100}cm"
    )

    # Localization recall
    localized = [
        100
        * np.sum(
            (np.array(metrics.t_err[pose_mask]) < t_th)
            & (np.array(metrics.R_err[pose_mask]) < R_th)
        )
        / n_queries
        for t_th, R_th in pose_thresholds
    ]
    _logger.info(
        "Localize recall@(<5cm5deg/0.25m2deg/0.5m5deg/1m10deg): "
        f"{localized[0]:.2f}/{localized[1]:.2f}/{localized[2]:.2f}/{localized[3]:.2f} %"
    )

    # Check distribution
    reproj_hists = check_data_hist(
        metrics.reproj_gt3d_estpose,
        bins=[0, 0.5, 1, 5, 10, 20, 100, 1000],
        tag=f"Reproj",
    )
    _logger.info(f"\n{reproj_hists}")

    # Compute reproj, recall AUC
    mean_reproj_errs = [np.mean(errs) for errs in metrics.reproj_gt3d_estpose]
    nsamples = len(mean_reproj_errs)

    # Add vals for failed samples
    mean_reproj_errs += [np.inf] * (n_queries - len(mean_reproj_errs))
    reproj_auc = 100 * cal_error_auc(mean_reproj_errs, auc_thresholds)
    _logger.info(
        f"\nReprojThres={auc_thresholds}px AUC={reproj_auc} nsamples={nsamples}"
    )

    # Runtime
    if "match_time" in metrics:
        sample_match_time = 1e3 * np.sum(metrics.match_time) / n_queries
        _logger.info(f"Matching time (per pair):{sample_match_time:.2f} ms")

    if return_qs:
        return (t_qs, R_qs)
    return reproj_auc
