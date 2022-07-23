from typing import List, Optional

import torch
import torch.nn as nn


def conv1d_layer(
    in_channel: int, out_channel: int, normalize: Optional[str] = "ins"
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Conv1d(in_channel, out_channel, kernel_size=1)]
    if normalize == "ins":
        layers.append(nn.InstanceNorm1d(in_channel))
    if normalize and "bn" in normalize:
        layers.append(
            nn.BatchNorm1d(in_channel, track_running_stats=(normalize == "bn_untrack"))
        )
    return nn.Sequential(*layers)


class conv1d_residual_block(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        mid_channel: Optional[int] = None,
        normalize: Optional[str] = "ins",
        activation: str = "relu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        mid_channel = out_channel if mid_channel is None else mid_channel
        self.preconv = conv1d_layer(
            in_channel=in_channel, out_channel=mid_channel, normalize=None
        )
        self.conv1 = conv1d_layer(
            in_channel=mid_channel, out_channel=mid_channel, normalize=normalize
        )
        self.conv2 = conv1d_layer(
            in_channel=mid_channel, out_channel=out_channel, normalize=normalize
        )
        self.act = nn.LeakyReLU() if "leaky" in activation else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = x
        x = self.preconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        if self.residual:
            x = x + x_residual
        return x


class PointResNet(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_layers: int = 12,
        feat_channel: int = 128,
        mid_channel: int = 128,
        activation: str = "relu",
        normalize: Optional[str] = "ins",
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        # First convolution
        self.conv_in = nn.Sequential(
            *[nn.Conv1d(in_channel, feat_channel, kernel_size=1)]
        )
        for i in range(self.num_layers):
            setattr(
                self,
                f"conv_{i}",
                conv1d_residual_block(
                    in_channel=feat_channel,
                    out_channel=feat_channel,
                    mid_channel=mid_channel,
                    normalize=normalize,
                    activation=activation,
                    residual=residual,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(self.num_layers):
            x = getattr(self, f"conv_{i}")(x)
        return x
