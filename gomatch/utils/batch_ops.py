from typing import Union, cast

import torch
from torch_scatter import scatter_sum


def batchify_b(
    pts: torch.Tensor, batch: torch.Tensor, value: Union[int, float, bool] = 0
) -> torch.Tensor:
    sizes = scatter_sum(torch.ones_like(batch), batch)
    out = torch.full(
        (len(sizes), sizes.max(), pts.size(-1)),
        value,
        dtype=pts.dtype,
        device=pts.device,
    )

    idx = torch.arange(len(batch), dtype=batch.dtype, device=batch.device)
    start_idx = torch.tensor(
        [0, *sizes[:-1]], dtype=torch.long, device=sizes.device
    ).cumsum(0)

    # assign where appropriate
    out[batch, idx - start_idx[batch]] = pts
    return out


def batchify_tile_b(pts: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    sizes = scatter_sum(torch.ones_like(batch), batch)
    size_max = sizes.max()
    out = torch.empty(
        (len(sizes), size_max, pts.size(-1)), dtype=pts.dtype, device=pts.device
    )

    for bid, size in enumerate(sizes):
        out[bid] = pts[bid == batch].repeat(
            cast(int, torch.ceil(size_max / size).int()), 1
        )[:size_max]
    return out


def flatten_b(pts: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    sizes = scatter_sum(torch.ones_like(batch), batch)
    idx = torch.arange(len(batch), dtype=batch.dtype, device=batch.device)
    start_idx = torch.tensor(
        [0, *sizes[:-1]], dtype=torch.long, device=sizes.device
    ).cumsum(0)

    # assign where appropriate
    return pts[batch, idx - start_idx[batch]]
