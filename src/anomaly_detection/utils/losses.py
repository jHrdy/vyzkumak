import torch

def wasserstein_1d_loss(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    """
    p, q: shape (batch_size, 96)
    assumes non-negative values
    """

    # normalize to probability distributions
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    # cumulative distribution functions
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)

    # W1 distance
    loss = torch.mean(torch.abs(cdf_p - cdf_q).sum(dim=-1))
    return loss