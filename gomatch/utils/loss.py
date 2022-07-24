from collections import defaultdict
from typing import Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn

from .geometry import project3d_normalized
from .metrics import io_metric


def compute_loss(
    data: Mapping[str, torch.Tensor],
    preds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    opt_inliers_only: bool = False,
    cls: bool = False,
    rpthres: float = 1,
) -> Dict[str, torch.Tensor]:
    # preds: ot_scores, *match_probs_b
    bids = torch.unique_consecutive(data["idx2d"])
    i = 0
    losses = defaultdict(list)
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

        # Load gt matches
        matches_gt = data["matches_bin"][i : i + total].view(n3d + 1, n2d + 1)
        loss = ot_scores_b[bid].new_tensor(0.0)

        # OT log loss
        scores = ot_scores_b[bid]
        if opt_inliers_only:
            matches_gt = matches_gt[:n3d, :n2d]
            scores = scores[:n3d, :n2d]
        loss_log = torch.mean(-scores[matches_gt].log())
        loss += loss_log
        losses["loss_log"].append(loss_log)

        # Classification loss
        if cls:
            device = loss.device
            K = data["K"][bid]
            R_gt = data["R"][bid]
            t_gt = data["t"][bid]
            pts2d = data["pts2d"][mask2d].to(device)
            pts3d = data["pts3d"][mask3d].to(device)

            # Compute reprojection err
            i3d, i2d = torch.where(match_probs_b[bid][:-1, :-1] > -1)
            kps3d, kps2d = pts3d[i3d], pts2d[i2d, :2]
            kps2d_proj = project3d_normalized(R_gt, t_gt, kps3d)
            reproj_err = (kps2d - kps2d.new_tensor(kps2d_proj)).norm(dim=1)

            # Balanced BCE loss
            cls_preds = match_probs_b[bid][i3d, i2d]
            pos_mask = (reproj_err < rpthres).float()

            # Measure here instead of inside metric for convience
            cls_metrics = io_metric(cls_preds > 0.5, reproj_err < rpthres)
            losses["cls_recall"].append(cls_metrics["recall"])
            losses["cls_prec"].append(cls_metrics["precision"])

            # Record reproj errors
            losses["matched_reproj_err"].append(reproj_err.mean())
            losses["cls_pos_reproj_err"].append(reproj_err[cls_preds > 0.5].mean())

            # Only when there are positive samples
            if pos_mask.sum() > 0:
                neg_mask = 1 - pos_mask
                pwei = neg_mask.sum() / pos_mask.sum() * pos_mask + neg_mask
                loss_cls = nn.functional.binary_cross_entropy(
                    cls_preds, pos_mask, reduction="none"
                )
                loss_cls = (pwei * loss_cls).mean()
            else:
                loss_cls = cls_preds.new_tensor(0.0)
            loss += loss_cls
            losses["loss_cls"].append(loss_cls)
            if len(i3d) > 0:
                losses["cls_pos_rate"].append(pos_mask.sum() / len(i3d))

        losses["loss"].append(loss)
        i += total
    mean_losses = {k: torch.mean(torch.stack(v)) for k, v in losses.items()}
    return mean_losses
