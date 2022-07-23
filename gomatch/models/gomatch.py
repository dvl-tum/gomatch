from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from .ot import RegularisedOptimalTransport, init_couplings_and_marginals
from .bpnpnet import pairwiseL2Dist
from .net_modules import PointResNetEncoder, SCAtt2D3D, MatchCls2D3D

from ..utils.extract_matches import mutual_assignment


class OTMatcher(nn.Module):
    def __init__(
        self,
        p3d_type: str,
        kp_feat_dim: int = 128,
        share_kp2d_enc: bool = True,
        att_layers: Iterable[str] = ("self", "cross", "self"),
    ) -> None:
        super().__init__()

        # 2D encoder
        self.kp2d_enc = PointResNetEncoder(in_channel=2, feat_channel=kp_feat_dim)

        # 3D encoder
        p3d_dim = 3 if p3d_type == "coords" else 2
        if share_kp2d_enc and p3d_dim == 2:
            self.kp3d_enc = self.kp2d_enc
        else:
            self.kp3d_enc = PointResNetEncoder(
                in_channel=p3d_dim, feat_channel=kp_feat_dim
            )

        # Initialize OT
        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.ot = RegularisedOptimalTransport()

        # Initialize Attention
        self.attention = SCAtt2D3D(att_layers) if att_layers else None

    def encode_pts(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        desc2d = self.kp2d_enc(pts2d, idx2d)
        desc3d = self.kp3d_enc(pts3d, idx3d)
        return desc2d, desc3d

    def ot_match(self, desc2d: torch.Tensor, desc3d: torch.Tensor) -> torch.Tensor:
        # Matching distances
        idists = pairwiseL2Dist(desc3d.transpose(-2, -1), desc2d.transpose(-2, -1))

        # Optimal transport
        cost, mu, nu = init_couplings_and_marginals(idists, bin_cost=self.bin_score)
        iscores = self.ot(cost, mu, nu).squeeze(0)
        return iscores

    def forward_sample(
        self,
        desc2d: torch.Tensor,
        desc3d: torch.Tensor,
        pts2d: torch.Tensor,
        pts3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # desc2d, desc3d:  (N, 2), (N, 3)
        n2, n3 = len(desc2d), len(desc3d)

        if self.attention is not None:
            desc2d, desc3d = self.attention(desc2d, desc3d, pts2d, pts3d)  # N, C

        # Reshape descriptors
        desc2d = desc2d.T.unsqueeze(0)  # B, C, N
        desc3d = desc3d.T.unsqueeze(0)

        # L2 Normalise
        desc2d = nn.functional.normalize(desc2d, p=2, dim=1)
        desc3d = nn.functional.normalize(desc3d, p=2, dim=1)

        # OT matching scores
        iscores = self.ot_match(desc2d, desc3d)
        return iscores, desc2d, desc3d

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> List[torch.Tensor]:
        # Keypoint Encoding
        desc2d, desc3d = self.encode_pts(
            pts2d,
            idx2d,
            pts3d,
            idx3d,
        )

        # Iterate each sample
        nb = len(torch.unique_consecutive(idx2d))
        scores_b = []
        for ib in range(nb):
            mask2d = ib == idx2d
            mask3d = ib == idx3d

            # Descriptor matching
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            idesc2d, idesc3d = desc2d[mask2d], desc3d[mask3d]
            iscores, _, _ = self.forward_sample(idesc2d, idesc3d, ipts2d, ipts3d)
            scores_b.append(iscores)
        return scores_b


class OTMatcherCls(nn.Module):
    def __init__(
        self,
        p3d_type: str,
        kp_feat_dim: int = 128,
        share_kp2d_enc: bool = True,
        att_layers: Iterable[str] = ("self", "cross", "self"),
    ) -> None:
        super().__init__()

        # OT feature matcher
        self.raw_matcher = OTMatcher(
            p3d_type,
            kp_feat_dim=kp_feat_dim,
            share_kp2d_enc=share_kp2d_enc,
            att_layers=att_layers,
        )

        # Classifier for outlier match rejection
        self.classifier = MatchCls2D3D(kp_feat_dim=kp_feat_dim)

    def classify_sample(
        self,
        ipts2d,
        ipts3d,
        idesc2d: torch.Tensor,
        idesc3d: torch.Tensor,
        iscores: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: ipts2d and ipts3d are not being used. Remove them.

        # Select mutual matches
        match_mask = torch.tensor(mutual_assignment(iscores))
        i3d, i2d = torch.where(match_mask[:-1, :-1])
        if len(i3d) == 1:
            # Duplicate matches to ensure correct dimensions
            i3d = i3d.expand(2)
            i2d = i2d.expand(2)

        # Match feature selection
        f3d = idesc3d[:, :, i3d]  # B, C, N
        f2d = idesc2d[:, :, i2d]

        # Predict inlier match probs
        probs = self.classifier(f2d, f3d)

        # Construct output
        match_probs = -1.0 * torch.ones_like(iscores).to(probs)
        match_probs[i3d, i2d] = probs
        return match_probs

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Keypoint Encoding
        desc2d, desc3d = self.raw_matcher.encode_pts(
            pts2d,
            idx2d,
            pts3d,
            idx3d,
        )

        # Iterate each sample
        nb = len(torch.unique_consecutive(idx2d))
        scores_b: List[torch.Tensor] = []
        match_probs_b: List[torch.Tensor] = []
        for ib in range(nb):
            mask2d = ib == idx2d
            mask3d = ib == idx3d

            # Predict raw matches
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            idesc2d, idesc3d = desc2d[mask2d], desc3d[mask3d]
            iscores, idesc2d, idesc3d = self.raw_matcher.forward_sample(
                idesc2d, idesc3d, ipts2d, ipts3d
            )
            scores_b.append(iscores)

            # Classify inlier/outlier matches
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            match_probs = self.classify_sample(
                ipts2d,
                ipts3d,
                idesc2d,
                idesc3d,
                iscores,
            )
            match_probs_b.append(match_probs)
        return scores_b, match_probs_b


class GoMatchBVs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = OTMatcherCls(
            p3d_type="bvs", share_kp2d_enc=True, att_layers=["self", "cross", "self"]
        )

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.matcher(pts2d, idx2d, pts3d, idx3d)


class GoMatchCoords(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = OTMatcherCls(
            p3d_type="coords",
            share_kp2d_enc=False,
            att_layers=["self", "cross", "self"],
        )

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.matcher(pts2d, idx2d, pts3d, idx3d)
