# Original code src: https://github.com/overlappredator/OverlapPredator/blob/main/models/gcn.py

from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


def square_distance(
    src: torch.Tensor, dst: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalized:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def get_graph_feature(
    coords: torch.Tensor, feats: torch.Tensor, k: int = 10
) -> torch.Tensor:
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    k = min(k, N - 1)  # There are cases the input data points are fewer than k
    dist = square_distance(coords.transpose(1, 2), coords.transpose(1, 2))
    idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True)[
        1
    ]  # [B, N, K+1], here we ignore the smallest element as it's the query itself
    idx = idx[:, :, 1:]  # [B, N, K]

    idx = idx.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)  # [B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1, 1, 1, k)

    feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)

    return feats_cat


class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, k: int = 10) -> None:
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(
            feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


def MLP(channels: Sequence[int], do_bn: bool = True) -> nn.Sequential:
    """Multi-layer perceptron"""
    n = len(channels)
    layers: List[nn.Module] = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class SCAttention(nn.Module):
    """Predator + SuperGlue Self-Cross Attention Implementation"""

    def __init__(
        self,
        layer_names: Iterable[str],
        num_head: int = 4,
        feature_dim: int = 128,
        k: int = 10,
    ) -> None:
        super().__init__()
        self.names = layer_names
        layers: List[nn.Module] = []
        for atten_type in layer_names:
            if atten_type == "self":
                layers.append(SelfAttention(feature_dim, k))
            else:
                layers.append(AttentionalPropagation(feature_dim, num_head))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        coords0: torch.Tensor,
        coords1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: descs [B, C, N] *coords: [B, D, N]
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                desc0 = desc0 + layer(desc0, desc1)
                desc1 = desc1 + layer(desc1, desc0)
            elif name == "self":
                desc0 = layer(coords0, desc0)
                desc1 = layer(coords1, desc1)
        return desc0, desc1
