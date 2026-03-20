import torch
import torch.nn as nn

class Wasserstein1DLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        # normalize to probability distributions
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)
        q = q / (q.sum(dim=-1, keepdim=True) + self.eps)

        # cumulative distribution functions
        cdf_p = torch.cumsum(p, dim=-1)
        cdf_q = torch.cumsum(q, dim=-1)

        # W1 distance
        loss = torch.mean(torch.abs(cdf_p - cdf_q).sum(dim=-1))
        return loss