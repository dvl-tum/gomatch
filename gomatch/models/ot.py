from typing import Optional, Tuple
import torch


class RegularisedOptimalTransport(torch.nn.Module):
    def __init__(
        self, max_iters: int = 20, thresh: float = 1e-6, eps: float = 0.1
    ) -> None:
        super().__init__()

        # Initialize params
        self.max_iters = max_iters
        self.thresh = thresh
        self.eps = eps

    def forward(
        self, cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        P = sinkhorn_log(
            cost,
            mu,
            nu,
            self.eps,
            self.max_iters,
            self.thresh,
        )
        return P


def sinkhorn_log(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 0.1,
    max_iters: int = 50,
    thresh: float = 1e-6,
    acc_factor: Optional[float] = None,
) -> torch.Tensor:
    """Sinkhorn algorithm for regularized optimal transport in log space.
    Codes are adapated from https://github.com/gpeyre/SinkhornAutoDiff.

    Args:
        cost: cost matrices, (b, m, n)
        mu, nu: row-wise & column-wise target marginals
        eps: regularization factor; eps -> 0, closer to original ot problem
        thresh: sinkhorn stopping criteria
        max_iters: maximal number of sinkhorn iterations
        accelerate: bool, specify True to accelerate the unbalanced transport
    Return:
        P: optimal transport plan, (b, m, n)
    """

    if acc_factor:
        # To accelerate unbalanced transport
        lam = 0.5 ** 2 / (0.5 ** 2 + eps)
        tau = -acc_factor

    def ave(u: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u_prev

    def M(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

    # Sinkhorn iterations
    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    log_mu, log_nu = torch.log(mu + 1e-8), torch.log(nu + 1e-8)
    for _ in range(max_iters):
        u_prev = u

        # accelerated unbalanced iterations
        if acc_factor:
            u = ave(u, lam * (eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u))
            v = ave(
                v,
                lam
                * (
                    eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1))
                    + v
                ),
            )
        else:
            # Fixed point updates
            u = eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u
            v = eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1)) + v

        # Stopping criteria
        err = (u - u_prev).norm(dim=-1).mean()
        if err < thresh:
            break

    # Transport plan P = diag(a)*K*diag(b)
    P = torch.exp(M(u, v))
    return P


def init_couplings_and_marginals(
    cost: torch.Tensor, bin_cost: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, m, n = cost.shape
    one = cost.new([1])
    couplings = cost

    # Append dustbins to couplings  (SuperGlue version)
    m_bins = bin_cost.expand(b, m, 1)
    n_bins = bin_cost.expand(b, 1, n)
    last_bin = bin_cost.expand(b, 1, 1)
    couplings = torch.cat(
        [torch.cat([cost, m_bins], -1), torch.cat([n_bins, last_bin], -1)], 1
    )

    # Uniform marginals with dustbin
    mu = torch.cat([one.expand(m), one * n]) / (m + n)
    nu = torch.cat([one.expand(n), one * m]) / (m + n)
    mu = mu.unsqueeze(0)  # 1, m
    nu = nu.unsqueeze(0)  # 1, n
    return couplings, mu, nu
