from argparse import Namespace
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .datasets import BaseDataset
from ..utils.logger import get_logger

_logger = get_logger(level="INFO", name="data_loader")


def collate(all_data: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    # Ignore samples with no pts
    data = []
    for d in all_data:
        if len(d["pts2d"]) > 0:
            data.append(d)
    if len(data) == 0:
        batched: Dict[str, Any] = dict(name=None)
        return batched

    # Batch data contents
    batched = dict(
        name=[d["name"] for d in data],
        pts2d=torch.cat([torch.from_numpy(d["pts2d"]) for d in data]),
        pts3d=torch.cat([torch.from_numpy(d["pts3d"]) for d in data]),
        pts2d_pix=torch.cat([torch.from_numpy(d["pts2d_pix"]) for d in data]),
        pts2dm=torch.cat([torch.from_numpy(d["pts2dm"]) for d in data]),
        pts3dm=torch.cat([torch.from_numpy(d["pts3dm"]) for d in data]),
        idx2d=torch.cat(
            [
                torch.full((len(d["pts2d"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        idx3d=torch.cat(
            [
                torch.full((len(d["pts3d"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        matches_bin=torch.cat(
            [torch.from_numpy(d["matches_bin"]).view(-1) for d in data]
        ),
        R=torch.stack([torch.from_numpy(d["R"]) for d in data]),
        t=torch.stack([torch.from_numpy(d["t"]) for d in data]),
        K=torch.stack([torch.from_numpy(d["K"]) for d in data]),
    )

    # Special data for multi-covis evaluation
    if "unmerge_mask" in data[0]:
        batched["unmerge_mask"] = [d["unmerge_mask"] for d in data]
        batched["idx3dm"] = torch.cat(
            [
                torch.full((len(d["pts3dm"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        )
    if "covis_ids" in data[0]:
        batched["covis_ids"] = [d["covis_ids"] for d in data]
    return batched


def init_data_loader(
    config: Namespace,
    split: str = "train",
    batch: int = 16,
    overfit: int = -1,
    outlier_rate: Union[float, Tuple[float, float], None] = None,
    npts: Union[int, Tuple[int, int], None] = None,
) -> DataLoader:
    is_training = "train" in split
    batch = batch if "batch" not in config else config.batch
    num_workers = 0 if "num_workers" not in config else config.num_workers
    _logger.info(
        f"Init data loader: split={split} training={is_training} batch={batch}..."
    )

    # Load dataset
    dataset: torch.utils.data.Dataset = BaseDataset(config, split=split)
    assert isinstance(dataset, BaseDataset)
    if outlier_rate is not None:
        dataset.outlier_rate = outlier_rate
    if npts is not None:
        dataset.npts = npts
    _logger.info(dataset)

    if overfit > 0:
        dataset, _ = torch.utils.data.random_split(
            dataset, [overfit, len(dataset) - overfit]
        )

    # Wrap data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=num_workers,
        collate_fn=collate,
        shuffle=is_training,
        drop_last=is_training,
    )
    return data_loader
