from typing import List, cast

from bpnpnet.model.model import pairwiseL2Dist, STN3d
from bpnpnet.model.yi2018cvpr.model import Net as FeatureExtractor
from bpnpnet.model.yi2018cvpr.config import get_config
import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from .ot import RegularisedOptimalTransport
from ..utils.batch_ops import batchify_b


class BlindPnP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config_2d, _ = get_config()
        self.config_3d, _ = get_config()
        self.config_2d.in_channel = 2
        self.config_2d.gcn_in_channel = 2
        self.config_3d.in_channel = 3
        self.config_3d.gcn_in_channel = 3
        self.stn = STN3d()
        self.FeatureExtractor2d = FeatureExtractor(self.config_2d)
        self.FeatureExtractor3d = FeatureExtractor(self.config_3d)

        # Initialize OT
        self.sinkhorn = RegularisedOptimalTransport()

    def forward(
        self,
        p2d: torch.Tensor,
        p3d: torch.Tensor,
        num_points_2d: torch.Tensor,
        num_points_3d: torch.Tensor,
    ) -> torch.Tensor:
        f2d = p2d
        f3d = p3d

        # Transform f3d to canonical coordinate frame:
        trans = self.stn(f3d.transpose(-2, -1))  # bx3x3
        f3d = torch.bmm(f3d, trans)  # bxnx3

        # Extract features:
        f2d = self.FeatureExtractor2d(f2d.transpose(-2, -1)).transpose(
            -2, -1
        )  # b x m x 128
        f3d = self.FeatureExtractor3d(f3d.transpose(-2, -1)).transpose(
            -2, -1
        )  # b x n x 128

        # L2 Normalize:
        f2d = torch.nn.functional.normalize(f2d, p=2, dim=-1)
        f3d = torch.nn.functional.normalize(f3d, p=2, dim=-1)

        # Compute pairwise L2 distance matrix:
        M = pairwiseL2Dist(f2d, f3d)

        # Sinkhorn:
        # Set replicated points to have a zero prior probability
        b, m, n = M.size()
        r = M.new_zeros((b, m))  # bxm
        c = M.new_zeros((b, n))  # bxn
        for i in range(b):
            r[i, : cast(int, num_points_2d[i])] = 1.0 / num_points_2d[i].float()
            c[i, : cast(int, num_points_3d[i])] = 1.0 / num_points_3d[i].float()
        P = self.sinkhorn(M, r, c)
        return P


class BPnPMatcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BlindPnP()

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> List[torch.Tensor]:
        p2d = batchify_b(pts2d[:, :2], idx2d)
        p3d = batchify_b(pts3d, idx3d)

        num_points_2d = scatter_sum(torch.ones_like(idx2d), idx2d)[:, None]
        num_points_3d = scatter_sum(torch.ones_like(idx3d), idx3d)[:, None]

        # return confidence matrix 2d x 3d
        C = self.model.forward(p2d, p3d, num_points_2d, num_points_3d)

        # Unify output format as gomatch
        out = []
        for i in range(len(p2d)):
            n2d = num_points_2d[i]
            n3d = num_points_3d[i]

            # Make it 3d x 2d
            Ci = C[i, :n2d, :n3d].T

            # Append dustbins that are always 0
            Cia = Ci.new_zeros(n3d + 1, n2d + 1)
            Cia[:n3d, :n2d] = Ci
            out.append(Cia)
        return out


class BPnPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = BPnPMatcher()

    def forward(
        self,
        pts2d: torch.Tensor,
        idx2d: torch.Tensor,
        pts3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> List[torch.Tensor]:
        return self.matcher(pts2d, idx2d, pts3d, idx3d)
