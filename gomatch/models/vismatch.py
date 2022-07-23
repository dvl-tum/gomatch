from typing import List
import torch
from torch import nn


class VisDescMatcher(nn.Module):
    def forward(
        self,
        desc2d: torch.Tensor,
        idx2d: torch.Tensor,
        desc3d: torch.Tensor,
        idx3d: torch.Tensor,
    ) -> List[torch.Tensor]:
        # Iterate each sample
        nb = len(torch.unique_consecutive(idx2d))
        scores_b = []
        for ib in range(nb):
            mask2d = ib == idx2d
            mask3d = ib == idx3d

            # Load descs
            idesc2d, idesc3d = desc2d[mask2d], desc3d[mask3d]

            # Nearest neighbour matching
            idesc2d = idesc2d / idesc2d.norm(dim=1, keepdim=True)
            idesc3d = idesc3d / idesc3d.norm(dim=1, keepdim=True)
            similarity = torch.einsum("id, jd->ij", idesc3d, idesc2d)

            # Construct output
            iscores = similarity.new_zeros(len(idesc3d) + 1, len(idesc2d) + 1)
            iscores[:-1, :-1] = similarity
            scores_b.append(iscores)
        return scores_b
