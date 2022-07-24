from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .point_resnet import PointResNet
from .attention import SCAttention
from ..utils.batch_ops import batchify_tile_b, flatten_b


class PointResNetEncoder(PointResNet):
    def forward(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # type: ignore
        # batchify with tiling
        points_b = batchify_tile_b(points, idx)
        out = super().forward(points_b.transpose(-1, -2)).transpose(-2, -1)
        out = flatten_b(out, idx)
        return out


class SCAtt2D3D(nn.Module):
    """Self and cross attention for 2D3D matching."""

    def __init__(self, att_layers: Iterable[str] = ("self", "cross", "self")) -> None:
        super().__init__()
        self.att = SCAttention(att_layers)

    def forward(
        self,
        desc2d: torch.Tensor,
        desc3d: torch.Tensor,
        pts2d: torch.Tensor,
        pts3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input descs: [N, C]   pts: [N, D]
        desc2d = desc2d.unsqueeze(0).permute(0, 2, 1)
        desc3d = desc3d.unsqueeze(0).permute(0, 2, 1)
        pts2d = pts2d.unsqueeze(0).permute(0, 2, 1)
        pts3d = pts3d.unsqueeze(0).permute(0, 2, 1)

        # Perform attention
        desc2d, desc3d = self.att(desc2d, desc3d, coords0=pts2d, coords1=pts3d)
        desc2d = desc2d.permute(0, 2, 1).squeeze()
        desc3d = desc3d.permute(0, 2, 1).squeeze()
        return desc2d, desc3d


class MatchCls2D3D(nn.Module):
    """Match classification for 2D3D matching."""

    def __init__(
        self, kp_feat_dim: int = 128, feat_dim: int = 128, num_layers: int = 4
    ) -> None:
        super().__init__()
        self.encoder = PointResNet(
            in_channel=kp_feat_dim * 2,
            num_layers=num_layers,
            feat_channel=feat_dim,
            mid_channel=feat_dim,
        )
        self.conv_out = nn.Conv1d(feat_dim, 1, 1)

    def forward(self, f2d: torch.Tensor, f3d: torch.Tensor) -> torch.Tensor:
        # f2d, f3d: (B, C, N)

        # Feature fusion
        mfeat = torch.cat([f2d, f3d], dim=1)  # B=1, C, N

        # Predict probs
        mfeat = self.encoder(mfeat)
        logits = self.conv_out(mfeat).squeeze()  # N,
        probs = torch.sigmoid(logits)
        return probs
